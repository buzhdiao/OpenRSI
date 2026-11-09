# 中文说明:本文件是 builder_core/samples 下的"样例方案"(sample solution),
# 作为 LLM 生成方案时的 few-shot 参考样例(填充 gen_solution_script 等 prompt 中的
# {solution_sample} 占位符),示范一条完整的机器学习竞赛解题管线:
# 加载数据 -> 特征工程 -> 缺失值填补 -> 类别编码 -> 特征选择 -> 集成建模交叉验证 ->
# 标准化 -> 预测 -> 按 sample_submission 格式写出提交文件。
# 关键设计:路径全部来自环境变量(PUBLIC_ROOT/SUBMISSION_PATH),与沙箱运行约定一致,
# 使样例代码可以直接在 OpenMLE-Gym 沙箱中真实执行验证。

# 作为示例方案的模块 docstring:供 LLM prompt 引用的样例实现
"""Sample solution implementation used as an LLM prompt reference."""
# os:读取 PUBLIC_ROOT/SUBMISSION_PATH 环境变量、拼接数据路径
import os
# warnings:压制告警输出,保持沙箱运行日志干净
import warnings
# numpy:数值数组操作(取列值数组等)
import numpy as np
# pandas:表格数据读取/变换/写出
import pandas as pd
# sklearn:分层交叉验证与交叉验证评分
from sklearn.model_selection import StratifiedKFold, cross_val_score
# sklearn:标签编码(类别列转整数)与标准化(线性模型需要)
from sklearn.preprocessing import LabelEncoder, StandardScaler
# sklearn:三种基学习器与软投票集成器
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# sklearn.linear_model:逻辑回归(线性基学习器)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
# re:正则抽取乘客称谓(title)
import re

# 全局忽略警告,避免运行日志被无关告警污染
warnings.filterwarnings('ignore')

def load_data(base_path):
    """加载训练集、测试集与提交样例三份 CSV。"""
    # 读训练集(含标签)
    train = pd.read_csv(os.path.join(base_path, 'train.csv'))
    # 读测试集(无标签)
    test = pd.read_csv(os.path.join(base_path, 'test.csv'))
    # 读提交样例(确定提交格式)
    sample_sub = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
    # 三份数据一起返回
    return train, test, sample_sub

def extract_title(name):
    """从乘客姓名中抽取称谓(title),并归并稀有称谓。"""
    # 正则匹配"空格+字母串+句点"形式的称谓(如 " Mr." / " Dr.")
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    # 匹配到才做归并
    if title_search:
        # 取捕获组的称谓词
        title = title_search.group(1)
        # 稀有贵族/军衔/职业称谓统一归为 Rare(降低类别稀疏度)
        if title in ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
            return 'Rare'
        # Mlle/Ms 语义等同 Miss,统一归类
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        # Mme 语义等同 Mrs,统一归类
        elif title == 'Mme':
            return 'Mrs'
        # 常见称谓原样返回
        return title
    # 正则未匹配到称谓:返回 Unknown 占位
    return 'Unknown'

def extract_cabin_deck(cabin):
    """从舱位号中抽取甲板字母(首字符)。"""
    # 舱位缺失:返回 Unknown
    if pd.isna(cabin):
        return 'Unknown'
    # 按空白拆分(一人可能有多段舱位号)
    cabins = str(cabin).split()
    # 取第一段的首字母作为甲板
    if len(cabins) > 0:
        return cabins[0][0]
    # 空字符串兜底:返回 Unknown
    return 'Unknown'

def extract_ticket_prefix(ticket):
    """抽取票号前缀(字母/数字混合的票头部分)。"""
    # 票号缺失:返回 Unknown
    if pd.isna(ticket):
        return 'Unknown'
    # 统一转字符串处理
    ticket = str(ticket)
    # 含空格说明有前缀:取首段并去掉点/斜杠,再转大写
    if ' ' in ticket:
        return ticket.split()[0].replace('.', '').replace('/', '').upper()
    # 无空格说明纯数字票号:返回 None 类别
    return 'None'

def engineer_features(df):
    """基于原始列派生新特征(特征工程)。"""
    # 拷贝副本,避免修改调用方原数据
    df = df.copy()
    
    # 从姓名派生称谓特征
    df['Title'] = df['Name'].apply(extract_title)
    
    # 家庭规模 = 同舱兄弟姐妹/配偶数 + 父母子女数 + 本人
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # 是否独自行(家庭规模为 1),转 0/1 整数
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 从舱位号派生甲板特征
    df['CabinDeck'] = df['Cabin'].apply(extract_cabin_deck)
    
    # 是否有舱位记录(有舱位通常代表更高社会等级),转 0/1
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    
    # 从票号派生票号前缀特征
    df['TicketPrefix'] = df['Ticket'].apply(extract_ticket_prefix)
    
    # 年龄分段(离散化分箱;填补缺失后会重新生成)
    df['AgeBand'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100], labels=['Child', 'Teen', 'Young', 'Adult', 'Senior'])
    
    # 人均票价 = 票价 / 家庭规模
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    # 性别二值化:男性记 1
    df['SexEncoded'] = (df['Sex'] == 'male').astype(int)
    
    # 舱等转字符串,便于后面统一做标签编码
    df['PclassStr'] = df['Pclass'].astype(str)
    
    # 返回带新特征的副本
    return df

def impute_missing_values(train, test):
    """对训练/测试集做缺失值填补(全部以训练集统计量为基准,防泄漏)。"""
    # 各自拷贝,避免原地污染
    train = train.copy()
    test = test.copy()
    
    # 用"称谓+舱等"分组的中位数填补年龄(同组人群年龄更接近)
    age_medians = train.groupby(['Title', 'Pclass'])['Age'].median()
    
    def fill_age(row):
        # 嵌套函数:按行填补年龄
        # 年龄缺失才需要填补
        if pd.isna(row['Age']):
            # 构造分组键
            key = (row['Title'], row['Pclass'])
            # 分组存在则用组中位数
            if key in age_medians:
                return age_medians[key]
            # 分组不存在(如测试集罕见组合)时退回全局中位数
            return train['Age'].median()
        # 不缺失原样返回
        return row['Age']
    
    # 对训练集逐行应用年龄填补
    train['Age'] = train.apply(fill_age, axis=1)
    # 对测试集逐行应用同一套填补逻辑
    test['Age'] = test.apply(fill_age, axis=1)
    
    # 票价缺失(测试集有):用"舱等+登船港"分组中位数填补
    fare_median = train.groupby(['Pclass', 'Embarked'])['Fare'].median()
    
    def fill_fare(row, df, is_train=True):
        # 嵌套函数:按行填补票价
        # 票价缺失才填补
        if pd.isna(row['Fare']):
            # 构造分组键
            key = (row['Pclass'], row['Embarked'])
            # 分组存在则用组中位数
            if key in fare_median:
                return fare_median[key]
            # 否则退回训练集全局中位数
            return train['Fare'].median()
        # 不缺失原样返回
        return row['Fare']
    
    # 只对测试集做票价填补(训练集票价无缺失;Embarked 为 NaN 时分组键会落空,故由兜底逻辑接管)
    test['Fare'] = test.apply(lambda r: fill_fare(r, test, False), axis=1)
    
    # 登船港缺失:用训练集众数填补
    embarked_mode = train['Embarked'].mode()[0]
    # 训练集填补
    train['Embarked'] = train['Embarked'].fillna(embarked_mode)
    # 测试集用同一众数填补
    test['Embarked'] = test['Embarked'].fillna(embarked_mode)
    
    # 返回填补后的两份数据
    return train, test

def encode_features(train, test):
    """对类别特征做标签编码(训练+测试合并拟合,保证编码空间一致)。"""
    # 各自拷贝
    train = train.copy()
    test = test.copy()
    
    # 需要标签编码的类别列清单(含派生列)
    categorical_cols = ['Sex', 'Embarked', 'Title', 'CabinDeck', 'TicketPrefix', 'AgeBand', 'PclassStr']
    
    # 逐列编码
    for col in categorical_cols:
        # 列存在才处理(不同数据集列可能缺失)
        if col in train.columns:
            # 每列独立创建一个标签编码器
            le = LabelEncoder()
            # 训练侧值统一转字符串并把缺失记为 Unknown
            train_vals = train[col].astype(str).fillna('Unknown')
            # 测试侧同样处理
            test_vals = test[col].astype(str).fillna('Unknown')
            # 训练+测试合并后再拟合,确保两侧类别到整数的映射一致
            combined = pd.concat([train_vals, test_vals], axis=0)
            # 在合并值上拟合编码器
            le.fit(combined)
            # 训练侧生成 "<列名>_Encoded" 编码列
            train[col + '_Encoded'] = le.transform(train_vals)
            # 测试侧生成对应编码列
            test[col + '_Encoded'] = le.transform(test_vals)
    
    # 返回编码后的两份数据
    return train, test

def select_features(df):
    """筛选用于建模的特征列(只保留实际存在的列)。"""
    # 目标特征清单:原始数值列 + 派生列 + 各类别列的编码列
    feature_cols = [
        'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
        'FamilySize', 'IsAlone', 'HasCabin', 'SexEncoded', 'FarePerPerson',
        'Embarked_Encoded', 'Title_Encoded', 'CabinDeck_Encoded', 
        'TicketPrefix_Encoded', 'AgeBand_Encoded', 'PclassStr_Encoded'
    ]
    
    # 只取数据框中实际存在的特征列(顺序保持清单顺序)
    available_features = [col for col in feature_cols if col in df.columns]
    # 返回仅含这些列的子集
    return df[available_features]

def train_model(X_train, y_train):
    """用三个基学习器做软投票集成,先交叉验证评估再全量拟合。"""
    
    # 基模型一:随机森林(200 棵树,限深 10,限制分裂/叶样本数,固定随机种子,多核并行)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 基模型二:梯度提升树(150 轮,限深 5,学习率 0.1)
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        random_state=42
    )
    
    # 基模型三:逻辑回归(正则强度 1.0,迭代上限 1000)
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    
    # 软投票集成:三个基模型按预测概率平均投票(通常优于硬投票)
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft'
    )
    
    # 分层 5 折交叉验证(打乱并固定种子,保证类别比例与可复现性)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # 以 accuracy 为指标跑交叉验证
    scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='accuracy')
    # 打印交叉验证均值与 2 倍标准差区间
    print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # 用全部训练数据拟合最终集成模型
    ensemble.fit(X_train, y_train)
    
    # 返回训练好的模型
    return ensemble

def main():
    """端到端主流程:从环境变量取路径,完成特征工程到提交文件写出。"""
    # 数据根目录来自 PUBLIC_ROOT 环境变量(沙箱注入),缺省为当前目录
    base_path = os.environ.get('PUBLIC_ROOT', '')
    
    # 加载训练/测试/提交样例三份数据
    train_df, test_df, sample_sub = load_data(base_path)
    # 打印两份数据的样本量,便于核对
    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # 暂存乘客 ID 列(特征工程后仍要回填到提交文件)
    train_ids = train_df['PassengerId'].values
    test_ids = test_df['PassengerId'].values
    
    # 暂存标签列(存活与否)
    y_train = train_df['Survived'].values
    
    # 训练/测试分别做特征工程
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # 缺失值填补
    train_df, test_df = impute_missing_values(train_df, test_df)
    
    # 填补完成后按补齐的年龄重新生成年龄分段(填补前生成的分段对缺失值无效)
    train_df['AgeBand'] = pd.cut(train_df['Age'], bins=[0, 12, 20, 40, 60, 100], labels=['Child', 'Teen', 'Young', 'Adult', 'Senior'])
    test_df['AgeBand'] = pd.cut(test_df['Age'], bins=[0, 12, 20, 40, 60, 100], labels=['Child', 'Teen', 'Young', 'Adult', 'Senior'])
    
    # 类别特征编码
    train_df, test_df = encode_features(train_df, test_df)
    
    # 特征选择:取训练/测试两侧的特征矩阵
    X_train = select_features(train_df)
    X_test = select_features(test_df)
    
    # 打印实际用到的特征列,便于日志审计
    print(f"Features used: {X_train.columns.tolist()}")
    # 打印两侧特征矩阵形状,确保列对齐
    print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 逻辑回归对量纲敏感:对全体特征做标准化(fit 用训练集,测试集只 transform,防泄漏)
    scaler = StandardScaler()
    # 训练侧拟合+变换
    X_train_scaled = scaler.fit_transform(X_train)
    # 测试侧仅用同一 scaler 变换
    X_test_scaled = scaler.transform(X_test)
    
    # 训练集成模型
    model = train_model(X_train_scaled, y_train)
    
    # 对测试集做预测
    predictions = model.predict(X_test_scaled)
    
    # 组装提交 DataFrame:乘客 ID + 预测标签
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions.astype(int)
    })
    
    # 对齐提交样例格式:固定列顺序并强制整型
    submission = submission[['PassengerId', 'Survived']]
    # 乘客 ID 转整数
    submission['PassengerId'] = submission['PassengerId'].astype(int)
    # 标签转整数
    submission['Survived'] = submission['Survived'].astype(int)
    
    # 提交文件输出路径来自 SUBMISSION_PATH 环境变量,缺省为当前目录下 submission.csv
    save_path = os.environ.get('SUBMISSION_PATH', 'submission.csv')
    # 写出 CSV(不带索引列)
    submission.to_csv(save_path, index=False)
    # 打印保存路径
    print(f"Submission saved to {save_path}")
    # 打印提交文件形状
    print(f"Submission shape: {submission.shape}")
    # 打印预测标签分布,便于检查是否极端偏斜
    print(f"Prediction distribution: {submission['Survived'].value_counts().to_dict()}")

# 脚本直接运行时执行主流程(被沙箱以 python xxx.py 方式调用)
if __name__ == "__main__":
    main()
