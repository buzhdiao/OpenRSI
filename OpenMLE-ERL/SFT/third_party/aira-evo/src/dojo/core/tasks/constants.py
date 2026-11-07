# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 常量：测试集适应度(Fitness)结果的键名（模型在测试集上的得分）
TEST_FITNESS = "test_fitness"

# 常量：验证集适应度的键名（用于搜索阶段比较解的优劣，不直接泄露测试信息）
VALIDATION_FITNESS = "validation_fitness"

# 常量：辅助评估信息的键名（附加的评估元数据，如奖牌、名次等）
AUX_EVAL_INFO = "aux_eval_info"

# 常量：解是否有效的键名（布尔值，表示提交是否通过校验）
VALID_SOLUTION = "valid_solution"

# 常量：代码执行输出的键名（终端输出、退出码、执行耗时等）
EXECUTION_OUTPUT = "execution_output"

# 常量：任务描述的键名（目标、约束、评估方式等说明文本）
TASK_DESCRIPTION = "task_description"

# 常量：有效性反馈的键名（评分器 grader 对提交给出的反馈文本）
VALID_SOLUTION_FEEDBACK = "valid_solution_feedback"
