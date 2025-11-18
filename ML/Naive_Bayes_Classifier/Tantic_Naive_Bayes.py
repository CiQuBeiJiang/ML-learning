import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

from my_tools.plotting import plot_confusion_matrix

filepath = '../Dataset/Tantic/train.csv'


def extract_title(name):
    """头衔提取"""
    # 扩展正则，匹配带连字符的头衔（如Deakins）、法语头衔（Mme, Mlle）
    title_search = re.search(' ([A-Za-z]+)(?:\.|,)', name)  # 匹配“字母+点号”或“字母+逗号”
    if title_search:
        title = title_search.group(1).strip()
        # 语义合并：将同义头衔统一
        title_mapping = {
            'Mme': 'Mrs',  # 法语“夫人”→英语“Mrs”
            'Mlle': 'Miss',  # 法语“小姐”→英语“Miss”
            'Ms': 'Miss',  # Ms统一为Miss
            'Sir': 'Mr',  # 爵士头衔归为普通男性尊称
            'Lady': 'Mrs',  # 女贵族头衔归为已婚女性
            'Don': 'Mr',  # 西班牙语尊称→Mr
            'Dona': 'Mrs'  # 西班牙语女性尊称→Mrs
        }
        return title_mapping.get(title, title)  # 有映射则替换，无则保留原头衔

    return "Unknown"  # 异常情况返回Unknown


def clean_titles(df):
    """清理头衔：合并稀有头衔，减少特征数量"""
    # 1. 先提取头衔
    df['Title'] = df['Name'].apply(extract_title)

    # 2. 统计头衔出现次数，定义“稀有头衔”（阈值可调整，如<10次）
    title_counts = df['Title'].value_counts()
    rare_titles = title_counts[title_counts < 10].index

    # 3. 稀有头衔统一合并为“Rare”
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')

    # 4. 删除异常的“Unknown”
    df = df[df['Title'] != 'Unknown']

    return df

def processing_data(file_path):
    """
    :param file_path:
    :return: df_processing
    """

    df = pd.read_csv(file_path)

    # 将"male"替换为1，"female"替换为0
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # 应用函数提取和清理头衔，创建新列
    df = clean_titles(df)  # 提取+清理头衔
    df.drop('Name', axis=1, inplace=True)

    # 按Title分组计算平均值，并用该平均值填充组内缺失值，fillna()方法已经过时
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.mean()))

    # 登船港口的缺失值很少，直接删除缺失值所在的行
    df.dropna(subset=['Embarked'], inplace=True)

    # 将SibSp和Parch聚类，剥离新特征家庭规模
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 包含乘客本人

    # # 对分类特征做独热编码（朴素贝叶斯对特征类型数据敏感）
    # df = pd.get_dummies(df, columns=['Embarked', 'Title'], drop_first=True)

    # 对Age分箱
    # 优化分箱逻辑
    df['Age_bin'] = pd.cut(
        df['Age'], bins=[0, 3, 12, 18, 60, 100],  # 新增0-3岁
        labels=[0, 1, 2, 3, 4]  # 对应：婴儿、儿童、青少年、成年、老年
    ).astype(int)

    # 对Fare分箱（按四分位数）
    df['Fare_bin'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3]).astype(int)

    # 加入分箱：1=独自旅行，2-4=小家庭，≥5=大家庭，效果不显著，遂止
    # df['FamilySize_bin'] = pd.cut(
    #     df['FamilySize'],
    #     bins=[0, 1, 4, float('inf')],  # 0-1（独自）、1-4（小家庭）、4以上（大家庭）
    #     labels=[0, 1, 2],  # 0=独自旅行，1=小家庭，2=大家庭
    #     include_lowest=True  # 包含左边界（确保FamilySize=1被归入0类）
    # ).astype(int)

    # 引入交互特征
    df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex'].astype(str)
    # 对交互特征做LabelEncoder编码
    cat_features = ['Embarked', 'Title', 'Pclass_Sex']

    # 将bool类型转化为int类型，因为bool类型处理不了？
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # 统一末尾删除，防止错误
    # 同时删除 PassengerId , Ticket（没用）和 cabin（缺失值太多） 列
    df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
    # 处理后删除原始连续特征
    df.drop(['Age', 'Fare'], axis=1, inplace=True)
    df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    # 显示数据所有列
    # pd.set_option('display.max_columns', None)
    #
    # print(df.head(5))
    # df.info()

    return df

# 引入LabelEncoder编码
def encode_categorical_features(train_df, test_df, cat_features):
    """
    批量对分类特征做 LabelEncoder 编码
    参数：
        train_df: 训练集（仅用它拟合编码器，避免数据泄露）
        test_df: 测试集
        cat_features: 需要编码的分类特征列表
    返回：
        train_encoded: 编码后的训练集
        test_encoded: 编码后的测试集
        label_encoders: 存储每个特征的编码器（方便后续解码查看）
    """

    label_encoders = {}  # 保存每个特征的编码器，后续可查看编码映射
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for feat in cat_features:  # 循环处理 cat_features 中的每个特征
        le = LabelEncoder()
        # 仅用训练集拟合编码器
        le.fit(train_encoded[feat])
        # 对训练集和测试集分别编码
        train_encoded[feat] = le.transform(train_encoded[feat])
        test_encoded[feat] = le.transform(test_encoded[feat])
        # 保存编码器，方便后续查看“原始类别→整数标签”的映射
        label_encoders[feat] = le

    return train_encoded, test_encoded, label_encoders

def train_predict_naive_bayes(df):
    """
    训练朴素贝叶斯模型并进行预测和评估

    参数:
        df: 处理后的DataFrame，包含特征和目标变量Survived
    返回:
        model: 训练好的模型
        metrics: 包含评估指标的字典
    """
    # 1. 划分特征(X)和目标变量(y)
    X = df.drop('Survived', axis=1)  # 特征矩阵
    y = df['Survived']  # 目标变量

    # 2. 划分训练集和测试集（8:2，固定随机种子）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 定义需要编码的分类特征列表（就是你关心的 cat_features）
    cat_features = ['Embarked', 'Title', 'Pclass_Sex']  # 包含新增的交互特征

    # 批量编码分类特征（核心使用 cat_features 的步骤）
    X_train_encoded, X_test_encoded, label_encoders = encode_categorical_features(
        X_train, X_test, cat_features
    )

    # 新增检测是否存在字符串类型特征
    if any(X_train_encoded.dtypes == 'object'):
        print("警告：编码后仍存在字符串类型特征！")
        print("字符串特征列：", X_train_encoded.select_dtypes(include='object').columns.tolist())
    else:
        print("编码验证通过，所有特征均为数值类型")

    # # 查看编码映射
    # for feat, le in label_encoders.items():
    #     print(f"\n{feat} 编码映射：{dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 3. 初始化并训练朴素贝叶斯模型
    # 特征多为离散值（分箱结果、独热编码），选择CategoricalNB更合适
    # alphas = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    # for alpha in alphas:
    #     model = CategoricalNB(alpha=alpha)
    #     scores = cross_val_score(model, X_train, y_train, cv=5, scoring='precision')  # 关注精确率
    #     print(f"alpha={alpha}，存活类平均精确率：{scores.mean():.4f}")

    model = CategoricalNB(alpha=0.1, fit_prior=True)
    model.fit(X_train_encoded, y_train)

    # 4. 在测试集上进行预测
    y_pred = model.predict(X_test_encoded)
    # y_pred_proba = model.predict_proba(X_test_encoded)  # 预测概率

    # 5. 模型评估
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)  # 精确率、召回率等
    conf_matrix = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = conf_matrix.ravel()

    # 整理评估指标
    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

    # 打印关键评估结果
    print(f"模型准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    return model, metrics,tn, fp, fn, tp


# processing_data(filepath)

if __name__ == "__main__":
    # 数据预处理
    df_processing = processing_data(filepath)
    # 训练和预测
    model, metrics,tn, fp, fn, tp= train_predict_naive_bayes(df_processing)
    # 导入工具包画图
    cm = np.array([[tn, fp], [fn, tp]])
    classes = ['Survived', 'Dead']
    plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix')


