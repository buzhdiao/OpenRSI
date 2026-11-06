# 引入 asyncio，用于等待被中止请求的重试间隔
import asyncio
# 引入 json，用于解析/生成 HTTP 响应体
import json

# 引入 FastAPI 的 JSON 响应类
from fastapi.responses import JSONResponse
# 引入 Starlette 中间件基类
from starlette.middleware.base import BaseHTTPMiddleware
# 引入请求类型标注
from starlette.requests import Request
# 引入响应类型
from starlette.responses import Response
# 引入 transformers 自动 tokenizer
from transformers import AutoTokenizer

# 引入异步 POST 工具（用于回查路由器接口）
from slime.utils.http_utils import post
# 引入按 loss mask 统计响应长度的工具
from slime.utils.mask_utils import get_response_lengths
# 引入样本数据结构
from slime.utils.types import Sample

# 引入本中间件的核心数据结构：字符串基数树（Radix Cache）
from .radix_tree import StringRadixTrie

# Hop-by-hop headers that should not be forwarded
# 逐跳头（hop-by-hop）集合：代理转发时必须剥离的 HTTP 头
HOP_BY_HOP = {
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
}


# 过滤掉不应转发的逐跳头
def _filter_headers(headers):
    """Filter out hop-by-hop headers that should not be forwarded."""
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


# 把流式响应安全地物化为普通响应对象（避免 Content-Length 不一致）
async def _materialize_response(resp):
    """Convert streaming-like Response into a regular Response/JSONResponse safely."""
    # Collect all bytes from the streaming response
    # 逐块收集流式响应的全部字节
    body = b""
    async for chunk in resp.body_iterator:
        body += chunk

    # Try to parse as JSON based on content-type
    # 取响应的内容类型与过滤后的头
    ct = resp.headers.get("content-type", "")
    headers = _filter_headers(resp.headers)

    # JSON 类型响应：尝试解析后重建 JSONResponse
    if "application/json" in ct:
        # If it's JSON, try to parse and return as JSONResponse
        try:
            data = json.loads(body.decode("utf-8"))
            return JSONResponse(content=data, status_code=resp.status_code, headers=headers)
        except Exception:
            # JSON parsing failed, fall back to raw bytes
            # 解析失败则按原始字节回退
            pass

    # Other types: return as raw bytes (without content-length)
    # 其他类型：按原始字节返回（去掉 content-length 由框架重算）
    return Response(content=body, status_code=resp.status_code, headers=headers, media_type=resp.media_type)


# 基数树中间件：拦截 /generate 请求，用 radix tree 提供前缀缓存并回写轨迹
class RadixTreeMiddleware(BaseHTTPMiddleware):
    # 初始化：绑定路由器并加载 tokenizer 与基数树
    def __init__(self, app, *, router):
        super().__init__(app)
        self.router = router
        self.args = router.args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
        self.radix_tree = StringRadixTrie(max_cache_size=10000, tokenizer=self.tokenizer, verbose=False)
        self.router.radix_tree = self.radix_tree

    # 请求分发入口：只处理 /generate，其余请求直接放行
    async def dispatch(self, request: Request, call_next):

        # 取请求路径
        path = request.url.path

        # 非 /generate 请求不做缓存处理
        if path != "/generate":
            return await call_next(request)

        # 解析请求 JSON 体
        request_json = await request.json()
        # 请求带文本时直接用
        if "text" in request_json:
            input_text = request_json.pop("text", "")
        # 请求带 token 序列时先解码为文本
        elif "input_ids" in request_json:
            input_text = self.tokenizer.decode(request_json["input_ids"])
        # 两者都没有：无法建立缓存
        else:
            input_text = None
        # 无有效输入文本：直接放行不做缓存
        if not input_text:
            return await call_next(request)
        # 从基数树取输入 token/logp/loss_mask（命中部分复用缓存）
        input_tokens, input_logprobs, input_loss_mask = self.radix_tree.retrieve_from_text(
            input_text, return_logprob=True
        )
        # 把 token 序列注入请求体，供下游按 token 处理
        request_json["input_tokens"] = input_tokens
        # 关闭流式（便于中间件整体解析响应）
        request_json["stream"] = False
        # 把修改后的 JSON 写回请求对象
        request._json = request_json

        # 解析出的响应数据（可能为 None）
        response_data = None
        # 最多重试 5 次（应对被中止的响应）
        for _ in range(5):
            response = await call_next(request)

            # If upstream returned a streaming response, materialize it to avoid Content-Length issues
            # 上游返回流式响应时物化为普通响应
            if response.__class__.__name__ == "_StreamingResponse":
                response = await _materialize_response(response)
            # Try to parse JSON from the current response for meta inspection
            # 尝试解析响应体 JSON 以检查完成原因
            try:
                if hasattr(response, "body") and isinstance(response.body, (bytes, bytearray)):
                    response_data = json.loads(response.body.decode("utf-8"))
                elif hasattr(response, "content") and isinstance(response.content, (dict, list)):
                    response_data = response.content  # JSONResponse.content is already a dict/list
            except Exception:
                response_data = None

            # 响应完整且非 abort 结束：跳出重试循环
            if (
                isinstance(response_data, dict)
                and "meta_info" in response_data
                and "finish_reason" in response_data["meta_info"]
                and response_data["meta_info"]["finish_reason"]["type"] != "abort"
            ):
                break
            # await 30 seconds for aborted responses
            # 被 abort 的响应：等待 30 秒后重试
            await asyncio.sleep(30)

        # 成功拿到含文本与输出 id 的完整响应时：把整条轨迹写回基数树缓存
        if isinstance(response_data, dict) and "text" in response_data and "output_ids" in response_data:
            generated_text = response_data["text"]

            full_text = input_text + generated_text
            if full_text:
                try:
                    # 服务端返回 token 级 logprob 时：精确缓存 token/logp/loss_mask
                    if "output_token_logprobs" in response_data.get("meta_info", {}):
                        generated_token_logprobs = [
                            item[0] for item in response_data["meta_info"]["output_token_logprobs"]
                        ]
                        generated_token_ids = [item[1] for item in response_data["meta_info"]["output_token_logprobs"]]
                        full_logprobs = input_logprobs + generated_token_logprobs
                        full_token_ids = input_tokens + generated_token_ids
                        full_loss_mask = input_loss_mask + [1] * len(generated_token_ids)
                        self.radix_tree.insert(
                            full_text,
                            full_token_ids,
                            full_logprobs,
                            full_loss_mask,
                            weight_version=response_data["meta_info"]["weight_version"],
                        )
                    else:
                        generated_token_ids = self.tokenizer(generated_text, add_special_tokens=False)["input_ids"]
                        full_token_ids = input_tokens + generated_token_ids
                        full_loss_mask = input_loss_mask + [1] * len(generated_token_ids)
                        self.radix_tree.insert(
                            full_text,
                            full_token_ids,
                            None,
                            full_loss_mask,
                            weight_version=response_data["meta_info"]["weight_version"],
                        )

                    if getattr(self.router, "verbose", False):
                        print(f"[slime-router] Successfully cached trajectory with {len(full_token_ids)} tokens")
                except Exception as e:
                    if getattr(self.router, "verbose", False):
                        print(f"[slime-router] Warning: Failed to cache trajectory: {e}")
        # 最终把响应返回给调用方
        return response


# 用 radix tree 后处理样本：从路由器回查整条轨迹的 token/logp/loss_mask
async def postprocess_sample_with_radix_tree(args, sample: Sample, output: dict):
    # slime router 暂不支持 partial rollout
    assert not args.partial_rollout, "Currently partial rollout is not supported when using slime router"
    # 路由器的文本回查接口地址
    retrieve_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/retrieve_from_text"
    # 回查载荷：完整文本（prompt + 生成部分），要求返回 logp
    retrieve_payload = {"text": sample.prompt + output["text"], "return_logp": True}
    # 异步回查路由器缓存
    retrieve_output = await post(retrieve_url, retrieve_payload)
    # 用回查结果直接覆盖样本 token 序列（免重新分词）
    sample.tokens = retrieve_output["tokens"]
    # 追加生成文本
    sample.response += output["text"]
    # 覆盖 loss mask（缓存中带前缀上下文的掩码）
    sample.loss_mask = retrieve_output["loss_mask"]
    # 按掩码统计响应长度
    sample.response_length = get_response_lengths([sample.loss_mask])[0]
    # 只保留响应段的 loss mask
    sample.loss_mask = sample.loss_mask[-sample.response_length :]
    # 只保留响应段的 rollout logprob
    sample.rollout_log_probs = retrieve_output["rollout_logp"][-sample.response_length :]
    # 返回后处理完成的样本
    return sample
