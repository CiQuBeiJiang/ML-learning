import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

filepath = '../Dataset/Tantic/train.csv'


# 1. 提取头衔（核心特征）
def extract_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    return title_search.group(1) if title_search else 'Unknown'


# 2. 数据预处理（只保留核心特征，拒绝冗余）
def processing_data(file_path):
    df = pd.read_csv(file_path)
    # 核心特征：性别、舱位、年龄、头衔、是否有Cabin、登船口
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})  # 性别编码
    df['Has_Cabin'] = df['Cabin'].notna().astype(int)  # 是否有Cabin
    df['Title'] = df['Name'].apply(extract_title)  # 提取头衔
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].where(df['Title'].isin(['Mr', 'Miss', 'Mrs', 'Master']), 'Rare')  # 合并稀有头衔
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('mean'))  # 年龄填充
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # 登船口填充

    # 独热编码（只对分类特征）
    df = pd.get_dummies(df, columns=['Title', 'Embarked'], drop_first=True)

    # 特征选择（只留6个核心特征）
    features = ['Pclass', 'Sex', 'Age', 'Has_Cabin', 'Title_Miss', 'Title_Mrs']
    df = df[features + ['Survived']]  # 确保包含目标变量

    # 标准化（年龄）
    scaler = StandardScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])

    return df


# 3. 训练模型（极简调优）
def train_model(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 超参数只调 C=1~10（逻辑回归的黄金区间）
    best_c = 1.0
    best_score = 0
    for c in [0.1, 0.5, 0.75, 1, 2, 5, 10]:
        model = LogisticRegression(C=c, penalty='l2', max_iter=1000, random_state=42)
        cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
        if cv_score > best_score:
            best_score = cv_score
            best_c = c

    # 训练最优模型
    model = LogisticRegression(C=best_c, penalty='l2', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"最优C={best_c}，测试集准确率：{acc:.4f}")
    print("\n特征权重（核心规律）：")
    for feat, weight in zip(X.columns, model.coef_[0]):
        print(f"{feat}: {weight:.3f}")  # 正权重=提升存活概率，负权重=降低

    return model, acc


if __name__ == "__main__":
    df = processing_data(filepath)
    model, acc = train_model(df)