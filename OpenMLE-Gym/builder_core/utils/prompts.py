# 中文说明:本模块是 OpenMLE-Gym builder_core 的 prompt 模板库,集中存放环境构建器
# (competition builder)在"把 Kaggle 风格原始竞赛数据加工为本地评测环境"的各阶段
# 调用 LLM 所用的全部提示词与填参占位符。
# 位置:builder_core/utils/prompts.py,被 builder_core 的构建管线导入。
# 关键设计:
# - 模板内 {xxx} 为 str.format 占位符,由构建器在运行期填充(如任务描述、样例);
#   因此字符串本体一个字符都不能改,格式化解析依赖这些占位符与 <code>/</code> 标记;
# - 多段 prompt 覆盖构建流水线各环节:文件盘点(perceiver)-> 任务描述生成 ->
#   prepare.py 生成 -> metric.py 生成 -> 方案脚本生成 -> 模态/算力分类 ->
#   进化阶段任务描述,各环节的输出被下一环节引用;
# - 生成代码的 prompt 统一要求 <code>...</code> 包裹,下游用
#   split("<code>")[1].split("</code>")[0] 提取,故模板中反复强调该约束。

# 文件盘点员 prompt:让 LLM 以"技术文件系统审计员"身份系统化盘点 raw/ 目录,
# 提取每个文件的元数据(行列数、表头、类型、前3行样例),不做主观解读,
# 最终通过 "save" 工具把盘点结果写入 fileinfo.txt,供后续生成任务描述使用
gen_perceiver = """
You are a Technical File System Auditor. Your sole purpose is to perform a systematic inventory of a directory, extracting raw metadata and factual content without interpretive bias.

## Core Objective
Exhaustively document the provided FILE TREE. For every file and directory identified, you must extract its content, and finally integrate your knowledge of the files and output a summary via the "save" tool.

## Operational Workflow
Think before you decide the tool use
Factual Extraction: 
- For Directories: List immediate children and sub-paths
- For CSVs: Use tool to retrieve exact row/column counts and a raw sample (first 3 rows)
- For txt/md/...: Use the tool once per path. A truncated preview is sufficient
  for inventory; never call the same text-reading tool on the same path again
  in an attempt to obtain the full file.
- For Other Files: Record filenames and extensions
When the inventory is enough and you decide to finish your work, call the "save" tool to write your final summary(your last tool call)
- The summary MUST be a factual, objective account of the files; avoid subjective deductions

## Output example for the "save" tool
Maintain a consistent, scannable format in "fileinfo.txt", make sure the file path(start from raw/) is complete and correct:

raw/core/train.csv
- Dimensions: [Rows] x [Cols]
- Headers: [List of column names]
- Data Types: [List of data type of each column]
- Raw Sample: [First 3 rows in JSON/Table format]
- Other Notes: ...
--------------------
raw/core/train/
- Contents: .png files
- Raw Sample: [3 .png file names]
- Other Notes: ...
--------------------
"""


# 任务描述生成 prompt:让 LLM 从盘点/原始材料中提取"机器学习编程"所需的关键信息,
# 按 {sample_description} 示例的格式输出结构化详细摘要(含评测细节),
# 并强调 raw/ 起始的文件路径必须完整正确(下游按此路径引用文件)
gen_description = """
You are an excellent expert in text information summarization. Your task is to extract key information for **MACHINE LEARNING PROGRAMMING** from given textual materials, and produce a structured, DETAILED summary in the same format as the EXAMPLE.
Make sure the Evaluation part covers all the details in the original content. 
Output Format Requirements: Your output MUST follow the format of the EXAMPLE, without any other explanatory text.
Make sure the file path(start from raw/) is **COMPLETE** and **CORRECT**.
## EXAMPLE:
{sample_description}
"""

# 组合上下文模板:把任务描述、数据集描述、本地文件树、文件内容四段拼装成
# 完整的任务说明上下文,供后续生成 prepare/metric/方案脚本时填充使用
description_info = """
## Description, Evaluation and Submission Instructions: 
{description}

## Dataset Description: 
{dataset_description}

## Local Raw File Tree: 
{tree_info}

## File Content (Combine this with the Dataset Description as the ### Files part):
{file_info}

"""

# prepare.py 生成 prompt:让 LLM 写出完整可运行的 prepare(raw, public, private) 脚本,
# 把 raw/ 原始数据重组为 data/public(训练侧可见)与 data/private(答案不可见)结构:
# 含标签数据 80/20 划分、图文音模态同步切分、防答案泄漏、生成 description.txt 与
# sample_submission.csv;输出同样要求 <code>...</code> 包裹以便机器提取
gen_prepare_script = """
You are a senior file organizer. Using the provided DESCRIPTION, write a complete, runnable `prepare.py` script to create a well-structured machine-learning competition environment from raw files.
It should implement EXACTLY "def prepare(raw: Path, public: Path, private: Path)".

## TARGET structure:
.
|-- data/
|   |-- public/
|   |   |-- train.csv (metadata/textual_data+labels for train_set), train/ (create only if there exists other concrete image/video/audio/..._data)
|   |   |-- test.csv (metadata/textual_data for test_set), test/ (create only if there exists other concrete image/video/audio/..._data)
|   |   |-- description.txt
|   |   |-- other metadata (*create only if there exists)
|   |   `-- sample_submission.csv
|   `-- private/ (invisible)
|       `-- test_answer.csv
`-- raw/ (preserve original)

The prepare function is a complete preparation process:
- Set deterministic behavior for complete train/test split process. 
    - Split raw data that CONTAINS LABELS into 80% train, 20% test (even if already split). 
    - Ignore raw data without labels (e.g. in most cases, the original test part should be ignored, while the train/val part can be aggregated): DO NOT use it for train/test split.
    - The image/audio/... files should also be split together with the CSV files into train/test folders.
    - Consider the correspondence between the test_answer and test_data. Ensure data/label alignment.
    - Don't include or leak anything related to answers/golden labels in "public/" directory.
    - Include necessary and comprehensive assert checks for the correctness of the split.
- Generate description.txt in public/ as an update of the original DESCRIPTION. 
    - Stay aligned with any changes made during preparation. 
    - Focus on revising the "Dataset Description" and "File" sections, like updating the size of the newly-split train/test set.
    - DO NOT mention the preparation details.
    - Mention the files in public/ ONLY!
- Generate sample_submission.csv from test_answer.csv. sample_submission.csv should have random but valid labels (same category as in test_answer.csv) rather than null values.
Finally (as is shown in the TARGET structure):
- "test_answer.csv" (participants shouldn't see) should be placed exactly in "private/" directory, while other files should be placed exactly in "public/" directory.
- "public/" directory MUST contain the "train.csv", "test.csv", "sample_submission.csv", "description.txt".

## Other REMINDERS:
- DO NOT alter /raw.
- You MUST NOT use symlink; always COPY the files from raw/.
- Rename files with names that might reveal their labels to avoid label leakage.
- Don't include data paths in CSV files.

## STRICT OUTPUT FORMAT:
- Your response MUST be composed of 2 parts: 
    - (1)Thinking Process: Start your response by analyzing the requirements and reflecting on any previous failed attempts. Ensure the logic is sound before proceeding to code generation.
    - (2)Pure Python Code: Provide the FINAL, COMPLETE, and RUNNABLE Python code. The code MUST be wrapped EXACTLY between <code> and </code> tags.
        - CRITICAL: The content inside the tags must contain ONLY valid Python code.
        - Ensure the closing </code> tag is present even if the code is long. 
      The CODE is later extracted using: `code = message.split("<code>")[1].split("</code>")[0].strip()` to retrieve your code. If you fail to provide these tags or include extra text inside them, the system will CRASH.

## EXAMPLE OF YOUR RESPONSE:
{prepare_sample}
"""

# metric.py 生成 prompt:让 LLM 按任务描述里的评测准则写出完整可运行的 metric.py,
# 主体必须是类名以 'Metrics' 结尾的类(评测器按此约定发现并实例化),
# 要处理 nan/inf 等数值边界,输出同样要求 <code>...</code> 包裹
gen_metric_script = """
You are a senior Python machine learning engineer. Write a complete, runnable `metric.py` script.

## Reminders:
- Must use the criteria presented in the description. Determine if it's 'higher_is_better'.
- Must work with sample_submission.csv and test_answer.csv.
- Follow the format strictly. The main body of the code MUST BE a class whose name ends with 'Metrics'.
- Deal with numerical values carefully to avoid nan/inf/etc.

## STRICT OUTPUT FORMAT:
- Your response MUST be composed of 2 parts: 
    - (1)Thinking Process: Start your response by analyzing the requirements. Ensure the logic is sound before proceeding to code generation.
    - (2)Pure Python Code: Provide the FINAL, COMPLETE, and RUNNABLE Python code. The code MUST be wrapped EXACTLY between <code> and </code> tags.
        - CRITICAL: The content inside the tags must contain ONLY valid Python code.
        - Ensure the closing </code> tag is present even if the code is long. 
      The CODE is later extracted using: `code = message.split("<code>")[1].split("</code>")[0].strip()` to retrieve your code. If you fail to provide these tags or include extra text inside them, the system will CRASH.
    
EXAMPLE OF YOUR RESPONSE:
{metric_sample}
"""

# 评测上下文模板:把"含评测准则的描述 + 提交样例 + 答案样例"拼给 metric 生成/校验环节,
# 让 LLM 明确评测口径与两份 CSV 的列结构
metric_info = """
Descriptions including Evaluation Criteria: 
{description}

Sample Submission CSV Format:
{submission_sample}

Test Answer CSV Format:
{answer_sample}

"""

# 附加信息模板:把模态、任务类型、文件大小、算力(CPU/GPU)等元信息追加到
# 任务描述末尾,供方案生成等环节参考
append_info = """

### Other Metadata

Modality: {modality}
Task-Type: {tasktype}
File Size: {filesize}
Using CPU/GPU: {cg}

"""

# 参赛方案生成 prompt:让 LLM 写出生产级 Python 方案脚本解决该 ML 任务,
# 硬性约束:数据根目录从 PUBLIC_ROOT 环境变量取、必须读 sample_submission.csv 对齐格式、
# 预测 CSV 写到 SUBMISSION_PATH、脚本自包含、自动检测 CUDA、禁 GUI/额外落盘文件、
# 固定保存逻辑、禁用 tensorflow;输出要求 <code>...</code> 包裹以便提取执行
gen_solution_script = """
## Task Overview
You are an expert machine learning engineer. Solve the following ML task with a production-ready Python script.
Description: 
{description}
        
## Environment & Path Constraints"
**Dataset Root**: All data files (train.csv, test.csv, images, etc.) are located at 'public/', Use exactly this logic to get the absolute path of the root:
```python
base_path = os.environ.get('PUBLIC_ROOT', '')
```
**Submissions**: You MUST read 'sample_submission.csv' from the dataset root to ensure the ID column and output format are identical.
**Output Path**: The final prediction CSV must be saved to the path provided by the environment variable 'SUBMISSION_PATH'.
    
## Coding Requirements
**Self-Contained**: Include all imports, class definitions, and 'if __name__ == \"__main__\":' block.
**Hardware**: Detect and use CUDA if available.",
**No GUI/Visuals**: DO NOT use plt.show() or any functions that open windows. Use 'matplotlib.use(\"Agg\")' if matplotlib is imported.
**No Extra Files**: Do NOT save any images, plots, or model checkpoints to the disk. Only output the final CSV.
**Saving Results**: Use exactly this logic:
```python
save_path = os.environ.get('SUBMISSION_PATH', 'submission.csv')
submission_df.to_csv(save_path, index=False)
```
**Library Use**: Never use 'tensorflow'!
        
## STRICT Format Instructions
- Your response MUST be composed of 2 parts: 
    - (1)Thinking Process: Start your response by analyzing the requirements. Ensure the logic is sound before proceeding to code generation.
    - (2)Pure Python Code: Provide the FINAL, COMPLETE, and RUNNABLE Python code. The code MUST be wrapped EXACTLY between <code> and </code> tags.
        - CRITICAL: The content inside the tags must contain ONLY valid Python code.
        - Ensure the closing </code> tag is present even if the code is long. 
      The CODE is later extracted using: `code = message.split("<code>")[1].split("</code>")[0].strip()` to extract your code. If you fail to provide these tags or include extra text inside them, the system will CRASH.

## EXAMPLE OF YOUR RESPONSE:
{solution_sample}
"""

# 模态/任务类型分类 prompt(两段式列表):
# 第[0]段要求分析描述并按 <out>模态@任务类型</out> 格式输出(可多模态、单任务类型),
# 如 Graph@Regression、Image@Classification;第[1]段是待填充的任务描述占位
gen_catg = ["""
Analyze the description and identify the machine learning task's modality and type. 
You may think before making decision, but the result must follow the format: [Modality]@[TaskType] and is wrapped in <out>, </out>.

Example formats:
- <out>Graph@Regression</out>
- <out>Text,Tabular@Classification</out>  
- <out>Image@Classification</out>
- <out>Video,Audio@Classification</out>

Available modalities (select one or multiple): Image, Text, Tabular, Audio, Video, Time-Series
Available task types (select one): Classification, Regression, Clustering, Segmentation, Object-Detection, Generation
""",
"Description: {description}"
]

# 算力判定 prompt(两段式列表):
# 第[0]段要求根据任务描述判断需要 GPU 还是仅 CPU 即可,答案(精确的 "GPU"/"CPU")
# 须包在 <out></out> 里便于解析;第[1]段填充任务描述并附文件大小作参考
gen_ifgpu = ["""
Based on the following machine learning task description, determine whether it requires a GPU or can run on a CPU alone. 
You may think before making decision, but make sure to wrap the answer(exactly "GPU" or "CPU") in <out>, </out>,
Which is <out>GPU</out> or <out>CPU</out>
""",
    "{description}\n File Size: {size}"
]

# 进化阶段任务描述 prompt:tts_search/进化搜索在"持续改进代码拿更高分"阶段使用,
# 只把任务描述填入 {description} 作为搜索迭代的上下文
evolve_description = """
You are an expert machine learning task solver. Read the following description of the ML task you are required to solve,
and then continuously optimize a Python code to obtain better metrics. Description:
{description}
"""
