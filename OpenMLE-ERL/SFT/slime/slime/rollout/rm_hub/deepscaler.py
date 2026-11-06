# 引入数学答案处理工具：\\boxed{} 答案抽取与 mathd/sympy 两种判分器
from .math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy


# DeepScaleR 规则奖励：从模型响应中提取最终答案并与标签比对，正确为 1 否则 0
def get_deepscaler_rule_based_reward(response, label):
    # 响应中含 </think> 结束标记时，取其后的正文作为解答部分
    if "</think>" in response:
        model_solution = response.split("</think>")[-1]
    # 否则若含 ###Response 分隔符，取其后的内容作为解答部分
    elif "###Response" in response:
        model_solution = response.split("###Response")[1]
    # 两种分隔符都不存在：无法定位解答，直接判 0 分
    else:
        return 0

    # 从解答部分抽取 \\boxed{} 内的最终答案
    model_answer = extract_answer(model_solution)
    # 抽取失败（没有可解析答案）：判 0 分
    if model_answer is None:
        return 0
    # 标签为空串：无参考答案可比，判 0 分
    if label == "":
        return 0

    # Convert single answer to list for uniform processing
    # 校验标签类型必须是字符串或数值
    assert isinstance(label, (str, float, int))
    # 把单个标签包装成列表，统一按多答案流程处理
    ground_truths = [label]

    # Process each ground truth
    # 预处理各参考答案（标签自身可能也带 \\boxed{} 包裹）
    processed_ground_truths = []
    # 逐个参考答案做规范化
    for truth in ground_truths:
        # 统一转成字符串
        truth = str(truth)
        # 标签内含 \\boxed 标记时从中抽取纯答案
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            # 抽取成功才纳入参考答案集合
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        # 不带 \\boxed 的标签原样作为参考答案
        else:
            processed_ground_truths.append(truth)

    # 预处理后没有任何可用参考答案：判 0 分
    if not processed_ground_truths:
        return 0

    # Check against all possible correct answers
    # 依次用 mathd 与 sympy 两种判分器与每个参考答案比对
    for ground_truth in processed_ground_truths:
        # 任一判分器判定相等即视为正确（两者互补以提高覆盖率）
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        # 比对成功：奖励 1
        if is_correct:
            return 1

    # 所有参考答案都不匹配：奖励 0
    return 0
