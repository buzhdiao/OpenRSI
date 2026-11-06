# 引入正则模块，用于文本规范化（去冠词）
import re
# 引入字符串常量模块，获取标点符号集合
import string
# 引入 Counter，用于词频统计（计算词重叠）
from collections import Counter


# 答案规范化：小写、去标点、去冠词、压缩空白，保证比对公平
def normalize_answer(s):

    # 去掉英文冠词 a/an/the（对语义无影响的虚词）
    def remove_articles(text):
        # 用词边界正则把冠词替换为空格
        return re.sub(r"\b(a|an|the)\b", " ", text)

    # 压缩连续空白为单个空格并去首尾
    def white_space_fix(text):
        # split 按任意空白切分，再用单空格拼接
        return " ".join(text.split())

    # 去除所有标点符号
    def remove_punc(text):
        # 构建标点字符集合
        exclude = set(string.punctuation)
        # 仅保留不在标点集合中的字符
        return "".join(ch for ch in text if ch not in exclude)

    # 转小写
    def lower(text):
        # 全部字符小写化
        return text.lower()

    # 按序执行：小写 -> 去标点 -> 去冠词 -> 压缩空白
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# 词级 F1 奖励：预测与参考答案之间的 token 级 F1/精确率/召回率（SQuAD 风格）
def f1_score(prediction, ground_truth):
    # 零分三元组常量：(f1, precision, recall) 均为 0
    ZERO_METRIC = (0, 0, 0)

    # 预测为 None：直接返回零分
    if prediction is None:
        return ZERO_METRIC

    # 规范化预测文本
    normalized_prediction = normalize_answer(prediction)
    # 规范化参考答案文本
    normalized_ground_truth = normalize_answer(ground_truth)

    # 预测是极简判定词（yes/no/noanswer）但与答案不一致：判零分
    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    # 答案是极简判定词但预测与之不一致：判零分
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    # 预测文本按空白切成 token 列表
    prediction_tokens = normalized_prediction.split()
    # 参考答案按空白切成 token 列表
    ground_truth_tokens = normalized_ground_truth.split()
    # Counter 取交集：得到两文本共同出现的词及重合次数（多重集合交）
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    # 重合词总次数
    num_same = sum(common.values())
    # 没有任何重合词：判零分
    if num_same == 0:
        return ZERO_METRIC
    # 精确率 = 重合词数 / 预测词数
    precision = 1.0 * num_same / len(prediction_tokens)
    # 召回率 = 重合词数 / 答案词数
    recall = 1.0 * num_same / len(ground_truth_tokens)
    # F1 = 精确率与召回率的调和平均
    f1 = (2 * precision * recall) / (precision + recall)
    # 返回 (f1, precision, recall) 三元组
    return f1, precision, recall
