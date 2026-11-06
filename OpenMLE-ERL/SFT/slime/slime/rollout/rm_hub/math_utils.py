# from https://github.com/agentica-project/deepscaler/blob/e6080ccd974eb64bd3430f0b36108244a6fee330/deepscaler/rewards/math_utils/utils.py
"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).
"""
# 引入正则模块，用于答案抽取与规范化
import re

# 引入 sympy，用于符号化简与数学等价判定
import sympy
# 引入 LaTeX 转文本工具，用于把 LaTeX 表达式转成 sympy 可读形式
from pylatexenc import latex2text
# 引入 sympy 表达式解析器
from sympy.parsing import sympy_parser


# Dan Hendrycks' code
# MATH 数据集（Dan Hendrycks）风格的答案规范化：统一格式便于字符串比对
def mathd_normalize_answer(answer: str | None) -> str | None:
    # 空答案原样返回 None
    if answer is None:
        return None
    # 去除首尾空白
    answer = answer.strip()
    # 规范化失败时退回原始答案
    try:
        # Remove enclosing `\text{}`.
        # 匹配整体形如 \text{...} 的包裹
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        # 命中则取 \text{} 内部内容
        if m is not None:
            answer = m.group("text").strip()
        # 进一步做深度字符串规范化
        return _strip_string(answer)
    # 任何异常都退回当前答案（判分宁可宽松不可崩溃）
    except Exception:
        return answer


# 深度字符串规范化：统一 LaTeX 分数/根号/单位/空格等书写差异
def _strip_string(string):
    # 修复简写分数：\frac1b / \frac12 -> \frac{1}{b} / \frac{1}{2}
    def _fix_fracs(string):
        # 按 \frac 切分各分数段
        substrs = string.split("\\frac")
        # 首段不含 \frac，直接保留
        new_str = substrs[0]
        # 存在分数段时逐个修复
        if len(substrs) > 1:
            # 跳过首段
            substrs = substrs[1:]
            # 逐段处理
            for substr in substrs:
                # 重新补回 \frac 前缀
                new_str += "\\frac"
                # 分数后已带花括号：原样保留
                if substr[0] == "{":
                    new_str += substr
                # 分数后无花括号：解析分子分母补全花括号
                else:
                    # 校验至少有 2 个字符可解析
                    try:
                        assert len(substr) >= 2
                    # 长度不足：放弃修复返回原串
                    except Exception:
                        return string
                    # 分子字符
                    a = substr[0]
                    # 分母起始字符
                    b = substr[1]
                    # 分母不带花括号的情形
                    if b != "{":
                        # 后面还有更多字符：分子分母都补花括号并接回剩余部分
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        # 恰好只有分子分母各一个字符
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    # 分母自带花括号的情形
                    else:
                        # 分母花括号后还有内容：分子补花括号，分母原样接回
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        # 分母花括号后无内容
                        else:
                            new_str += "{" + a + "}" + b
        # 用修复后的串替换原串
        string = new_str
        # 返回修复结果
        return string

    # 修复 a/b 形式的简单分数：转为 \frac{a}{b}
    def _fix_a_slash_b(string):
        # 不是恰好一段斜杠则不处理
        if len(string.split("/")) != 2:
            return string
        # 斜杠左侧为分子
        a = string.split("/")[0]
        # 斜杠右侧为分母
        b = string.split("/")[1]
        # 仅当分子分母都是纯整数时转换
        try:
            # 尝试转整数
            a = int(a)
            b = int(b)
            # 确保原文就是规范形式（无多余零等）
            assert string == f"{a}/{b}"
            # 组装成 \frac{a}{b}
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            # 返回转换结果
            return new_string
        # 任一步失败：维持原串
        except Exception:
            return string

    # 去除右侧单位描述（\text{ 开头的单位说明）
    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        # 含单位描述时截掉该部分
        if "\\text{ " in string:
            # 按 \text{ 切分
            splits = string.split("\\text{ ")
            # 校验只出现一次
            assert len(splits) == 2
            # 取前半部分（去掉单位）
            return splits[0]
        # 无单位描述则原样返回
        else:
            return string

    # 修复简写根号：\sqrt3 -> \sqrt{3}
    def _fix_sqrt(string):
        # 不含根号则不处理
        if "\\sqrt" not in string:
            return string
        # 按 \sqrt 切分
        splits = string.split("\\sqrt")
        # 首段直接保留
        new_string = splits[0]
        # 逐段补全花括号
        for split in splits[1:]:
            # 根号后无花括号：取首个字符为被开方数并补全
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            # 已带花括号则原样接回
            else:
                new_substr = "\\sqrt" + split
            # 累积拼接结果
            new_string += new_substr
        # 返回修复结果
        return new_string

    # linebreaks
    # 删除所有换行符
    string = string.replace("\n", "")

    # remove inverse spaces
    # 删除 LaTeX 负间距命令 \!
    string = string.replace("\\!", "")

    # replace \\ with \
    # 双反斜杠统一为单反斜杠
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    # 行间/文本分数命令统一为 frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    # 删除自适应定界符命令
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    # 删除角度符号 ^{\circ} 与 ^\circ
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    # 删除转义的美元符号
    string = string.replace("\\$", "")

    # remove units (on the right)
    # 去除右侧单位描述
    string = _remove_right_units(string)

    # remove percentage
    # 删除两种写法的百分号
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    # 补齐小数点前的 0（统一小数书写）
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    # 空串直接返回
    if len(string) == 0:
        return string
    # 首字符是小数点时前补 0
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    # 形如 "k = 值" 且变量名不超过 2 字符时：只保留等号右侧的值
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    # 修复简写根号
    string = _fix_sqrt(string)

    # remove spaces
    # 删除所有空格
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    # 修复简写分数与斜杠分数
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    # 特例：0.5 统一写成 \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    # 简单的 X/Y 输出统一转换为分数形式
    string = _fix_a_slash_b(string)

    # 返回规范化结果
    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
# 含这些子串的表达式易让 sympy 卡死，直接拒绝求值
BAD_SUBSTRINGS = ["^{", "^("]
# 连续幂指等可疑模式（正则），同样拒绝求值
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
# 元组/区间使用的括号字符集合
TUPLE_CHARS = "()[]"


# 用 sympy 解析数学表达式字符串（^ 转为 ** 幂运算）
def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    # 把 ^ 幂符号替换为 Python 的 ** 
    py_expr = expr.replace("^", "**")
    # We allow basic SymPy names but no builtins to prevent arbitrary code execution
    # This whitelist approach ensures SymPy's internal transformations (like Integer) still work
    # 构建安全符号表：仅允许 sympy 自身的公开名字，禁用内建函数防任意代码执行
    safe_dict = {k: v for k, v in sympy.__dict__.items() if not k.startswith("_")}
    # 调用 sympy 解析器：空局部字典 + 白名单全局字典，并启用隐式乘法等标准变换
    return sympy_parser.parse_expr(
        py_expr,
        local_dict={},
        global_dict={"__builtins__": {}, **safe_dict},
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


# 把 LaTeX 表达式转成 sympy 能读的普通表达式文本
def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    # 行间分数命令统一
    expr = expr.replace("\\tfrac", "\\frac")
    # 文本分数命令统一
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    # 用 pylatexenc 把 LaTeX 源码转成纯文本
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    # 把转换产生的特殊 Unicode 字符替换为 sympy 可识别的写法
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    # 去首尾空白后返回
    return expr.strip()


# 判断字符串能否解析为浮点数
def _is_float(num: str) -> bool:
    # 尝试转 float
    try:
        float(num)
        # 成功则返回 True
        return True
    # 失败返回 False
    except Exception:
        return False


# 判断浮点数是否（在误差范围内）等于某个整数
def _is_int(x: float) -> bool:
    # 与最近整数的距离不超过 1e-7 视为整数
    try:
        return abs(x - int(round(x))) <= 1e-7
    # 异常时视为非整数
    except Exception:
        return False


# 判断表达式是否为分数形式 a/b（分母非零）
def _is_frac(expr: str) -> bool:
    # 匹配可选负号、分子、斜杠、非零开头的分母
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


# 判断字符串是否表示一个整数值（容忍千分位逗号）
def _str_is_int(x: str) -> bool:
    # 解析失败视为非整数
    try:
        # 先去除规范格式的千分位逗号
        x = _strip_properly_formatted_commas(x)
        # 转浮点
        x = float(x)
        # 与最近整数误差极小则视为整数
        return abs(x - int(round(x))) <= 1e-7
    # 任一步异常返回 False
    except Exception:
        return False


# 把整数字符串安全转为 int（先去逗号）
def _str_to_int(x: str) -> int:
    # 删除所有逗号
    x = x.replace(",", "")
    # 先转浮点再取整
    x = float(x)
    # 返回整数
    return int(x)


# 注入带分数的隐式加法：如 7 3/4 -> 7+3/4，使其可求值
def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    # 匹配"数字 空格 数字"模式的编译正则
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    # 返回注入后的表达式
    return step


# 仅去除规范格式的千分位逗号（避免误伤元组逗号）
def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    # 匹配"数字+逗号+三位数字+结尾/非数字"的模式
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    # 循环替换直到不再变化（处理多个逗号）
    while True:
        # 执行一次替换
        next_expr = p1.sub("\\1\\3\\4", expr)
        # 无变化说明已替换完
        if next_expr == expr:
            break
        # 继续下一轮
        expr = next_expr
    # 返回最终结果
    return next_expr


# 答案表达式统一规范化：去外壳/单位/百分号/大小写等，产出可比较的规范串
def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    # 空输入返回 None
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    # 去掉整体包裹的 \text{...}
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    # 命中则取内部内容
    if m is not None:
        expr = m.group("text")

    # 以下为一系列等价写法归一
    # 转义百分号还原
    expr = expr.replace("\\%", "%")
    # 转义美元符还原
    expr = expr.replace("\\$", "$")
    # 删除美元符
    expr = expr.replace("$", "")
    # 删除百分号
    expr = expr.replace("%", "")
    # "or" 连接词统一为逗号分隔
    expr = expr.replace(" or ", " , ")
    # "and" 连接词统一为逗号分隔
    expr = expr.replace(" and ", " , ")

    # 数量级词统一转成 10 的幂
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    # 逐个删除常见单位词（含复数/(es)形式与次数幂）
    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        # 用正则删除单位及其可能的复数/幂形式
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    # 删除角度符号 ^\circ
    expr = re.sub(r"\^ *\\circ", "", expr)

    # 整体被花括号包裹时剥掉外壳
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    # 删除 ,\! 类分隔符残余
    expr = re.sub(",\\\\! *", "", expr)
    # 可解析为整数浮点时统一为整数字符串
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    # 仍含 LaTeX 命令时尝试转成 sympy 可读文本（失败则保持原样）
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    # edge case with mixed numbers and negative signs
    # 负号与空格的特殊情形：统一 "- " 为 "-"
    expr = re.sub("- *", "-", expr)

    # 注入带分数隐式加法
    expr = _inject_implicit_mixed_number(expr)
    # 删除所有空格
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    # 残余的 LaTeX 花括号直接丢弃
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    # 文本答案统一小写（大小写不敏感比较）
    expr = expr.lower()

    # 表示整数值的串统一为规范整数字符串
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    # 返回规范化结果
    return expr


# 统计表达式中未知函数/变量的字母种类数（用于求值安全性检查）
def count_unknown_letters_in_expr(expr: str):
    # sqrt 是已知函数名，先剔除
    expr = expr.replace("sqrt", "")
    # frac 是已知命令名，先剔除
    expr = expr.replace("frac", "")
    # 收集剩余的字母字符集合
    letters_in_expr = set([x for x in expr if x.isalpha()])
    # 返回字母种类数
    return len(letters_in_expr)


# 求值前安全检查：拒绝未知文本、多变量函数与可疑幂模式
def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    # 未知字母超过 2 个：可能是自然语言或高元函数，拒绝
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    # 含已知危险子串则拒绝
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    # 命中危险正则模式则拒绝
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    # 通过检查：允许求值
    return True


# 用 sympy 判定两个规范化表达式是否数学等价（差化简为 0）
def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    # 默认不等
    are_equal = False
    # 求值异常一律按不等处理
    try:
        # 构造差表达式：(答案)-(预测)
        expr = f"({ground_truth_normalized})-({given_normalized})"
        # 仅对安全表达式求值
        if should_allow_eval(expr):
            # 用 sympy 解析差表达式
            sympy_diff = _sympy_parse(expr)
            # 符号化简
            simplified = sympy.simplify(sympy_diff)
            # 差为 0 即两答案等价
            if simplified == 0:
                are_equal = True
    # 任何异常都不中断判分流程
    except Exception:
        pass
    # 返回等价判定结果
    return are_equal


# 切分元组/区间字符串为元素列表（容忍大数字中的千分位逗号）
def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    # 先剔除规范格式的千分位逗号
    expr = _strip_properly_formatted_commas(expr)
    # 空表达式返回空列表
    if len(expr) == 0:
        return []
    # 外层是成对括号且内部不含其他括号字符时：按逗号切分
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        # 去掉外壳括号后按逗号切分并去空白
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    # 不构成元组/区间：整体作为单元素
    else:
        elems = [expr]
    # 返回元素列表
    return elems


# 抽取字符串中最后一个 \\boxed 或 \\fbox 表达式（含外壳）
def last_boxed_only_string(string):
    # 从右向左查找 \\boxed
    idx = string.rfind("\\boxed")
    # 找不到时退而查找 \\fbox
    if idx < 0:
        idx = string.rfind("\\fbox")
        # 两者都没有：返回 None
        if idx < 0:
            return None

    # 扫描游标
    i = idx
    # 配对右括号位置（未找到为 None）
    right_brace_idx = None
    # 未闭合左括号层数计数
    num_left_braces_open = 0
    # 逐字符向右扫描
    while i < len(string):
        # 遇左括号层数加一
        if string[i] == "{":
            num_left_braces_open += 1
        # 遇右括号层数减一
        if string[i] == "}":
            num_left_braces_open -= 1
            # 层数归零即找到配对右括号
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        # 游标右移
        i += 1

    # 未找到配对右括号返回 None
    if right_brace_idx is None:
        retval = None
    # 否则返回完整表达式片段
    else:
        retval = string[idx : right_brace_idx + 1]

    # 返回抽取结果
    return retval


# 去掉 \\boxed{} 外壳（失败返回 None 而不抛异常）
def remove_boxed(s):
    # boxed 前缀常量
    left = "\\boxed{"
    # 校验外壳完整性并取内部内容
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    # 外壳不完整：返回 None
    except Exception:
        return None


# 从解答中抽取 \\boxed{} 内的最终答案
def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    # 先取最后一个 boxed 表达式
    solution = last_boxed_only_string(solution)
    # 再去外壳取内部内容
    solution = remove_boxed(solution)
    # 返回答案
    return solution


# sympy 判分器：规范化后先做字符串比较，再做元组逐元素等价判定
def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    # 规范化标签
    ground_truth_normalized = _normalize(ground_truth)
    # 规范化预测
    given_normalized = _normalize(given_answer)

    # 标签规范化失败：直接判错
    if ground_truth_normalized is None:
        return False

    # 规范化后字符串相等：直接判对
    if ground_truth_normalized == given_normalized:
        return True

    # 预测为空串：判错
    if len(given_normalized) == 0:
        return False

    # 切分标签的元组/区间元素
    ground_truth_elems = split_tuple(ground_truth_normalized)
    # 切分预测的元组/区间元素
    given_elems = split_tuple(given_normalized)

    # 标签是多元素但两答案的外层括号类型不一致：判错
    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0] or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    # 元素个数不一致：判错
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    # 逐元素比较
    else:
        # 配对遍历各元素
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems, strict=False):
            # 双方都是分数形式时
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                # 分数必须约分后完全一致（不允许 sympy 化简放水）
                is_correct = ground_truth_elem == given_elem
            # 一方是整数另一方不是时
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                # 标签为整数要求严格匹配（不允许 2.0 之类绕过）
                is_correct = False
            # 其余情形用 sympy 判定数学等价
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            # 任一元素不等即整体判错
            if not is_correct:
                break

    # 返回判分结果
    return is_correct


# mathd 判分器：按 MATH 数据集规范做字符串级严格比较
def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    # 规范化标签
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    # 规范化预测
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    # 规范化后字符串完全一致即判对
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    # 不一致判错
    return False


# 从文本中抽取答案：仅当含 \\boxed 时抽取，否则 None
def extract_answer(passage: str) -> str:
    # 文本含 \\boxed 标记才抽取
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    # 无 boxed：返回 None
    return None


# verl 风格综合判分入口：抽取双方答案后用 mathd 与 sympy 双判分器（任一通过即对）
def grade_answer_verl(solution_str, ground_truth):
    # 空标签直接判错
    if not ground_truth:
        return False
    # 标签统一转字符串
    ground_truth = str(ground_truth)
    # 标签自带 \\boxed 时先抽取内部答案
    if "\\boxed" in ground_truth:
        ground_truth = extract_answer(ground_truth)
    # 从模型解答中抽取 boxed 答案
    given_answer = extract_answer(solution_str)
    # 解答中无答案：判错
    if given_answer is None:
        return False
    # mathd 或 sympy 任一判分器通过即判对
    return grade_answer_mathd(given_answer, ground_truth) or grade_answer_sympy(given_answer, ground_truth)
