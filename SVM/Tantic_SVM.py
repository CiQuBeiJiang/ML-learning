import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from my_tools.plotting import plot_confusion_matrix
matplotlib.use('TkAgg')


class BasicSVM:
    def __init__(self, Kernel='rbf', C=1, gamma='auto'):
        self.scaler = MinMaxScaler()
        self.model = SVC(kernel=Kernel, C=C, gamma=gamma, probability=True)
        self.onehot_columns = None
        self.cabin_columns = None

    def preprocess_data(self, df, is_training=True):
        # 数据预处理方法
        # 缺失值处理
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0] if is_training else 'S')

        # 训练集Fare有缺失
        if is_training:
            self.fare_median = df['Fare'].median()
        df['Fare'] = df['Fare'].fillna(self.fare_median)

        # 年龄缺失值处理
        if is_training:
            age_medians = df.groupby(['Pclass', 'Sex'])['Age'].median()
            self.age_medians = age_medians
        df['Age'] = df.apply(lambda row: self.age_medians.loc[row['Pclass'], row['Sex']]
        if pd.isnull(row['Age']) else row['Age'], axis=1)

        # Cabin处理
        df['Cabin'] = df['Cabin'].fillna('U')
        df['Cabin_letter'] = df['Cabin'].str[0]

        # 特征缩放 - 仅对训练数据拟合
        if is_training:
            self.scaler.fit(df[['Age', 'Fare']])
        df[['Age', 'Fare']] = self.scaler.transform(df[['Age', 'Fare']])

        # 标签编码
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

        # 独热编码
        df = pd.get_dummies(df, columns=['Embarked', 'Pclass'], drop_first=True)

        # 确保测试集与训练集有相同的独热编码列
        if is_training:
            self.onehot_columns = df.columns
        else:
            missing_cols = set(self.onehot_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            df = df[self.onehot_columns]

        # Cabin独热编码
        df = pd.get_dummies(df, columns=['Cabin_letter'], drop_first=True)

        # 确保测试集与训练集有相同的Cabin编码列
        if is_training:
            self.cabin_columns = df.columns
        else:
            missing_cabin_cols = set(self.cabin_columns) - set(df.columns)
            for col in missing_cabin_cols:
                df[col] = 0
            df = df[self.cabin_columns]

        # 删除不需要的列
        X = df.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')

        if is_training:
            y = df['Survived']
            return X, y
        return X

    def train(self, X, y):
        # 划分数据集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 训练模型
        self.model.fit(X_train, y_train)

        # 预测
        y_pred = self.model.predict(X_val)

        # 输出评估指标
        print(f"验证集准确率：{accuracy_score(y_val, y_pred):.4f}")
        # 画出混淆矩阵
        cm = confusion_matrix(y_val, y_pred)
        classes = ['True','False']
        plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix')

        print("\n分类报告:\n", classification_report(y_val, y_pred))

    def evaluate(self, X, y):
        """评估模型在测试集上的表现"""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def predict(self, X):
        """预测新数据"""
        return self.model.predict(X)


# 加载训练数据
train_df = pd.read_csv("../Tantic/train.csv")

# 初始化模型
svm = BasicSVM(Kernel='rbf', C=1, gamma='auto')

# 预处理训练数据并训练模型
X_train, y_train = svm.preprocess_data(train_df)
svm.train(X_train, y_train)

# 加载并预处理测试数据
test_df = pd.read_csv("../Tantic/test.csv")
X_test = svm.preprocess_data(test_df, is_training=False)

# 如果测试集有标签，可以评估模型
if 'Survived' in test_df.columns:
    y_test = test_df['Survived']
    accuracy = svm.evaluate(X_test, y_test)
    print(f"测试集准确率：{accuracy:.4f}")
else:
    # 否则直接预测
    predictions = svm.predict(X_test)