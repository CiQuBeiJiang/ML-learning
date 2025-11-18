import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from my_tools.plotting import plot_confusion_matrix


class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.max_iter):
            updated = False
            for idx, x_i in enumerate(X):
                if y_[idx] * (np.dot(x_i, self.weights) + self.bias) <= 0:
                    self.weights += self.learning_rate * y_[idx] * x_i
                    self.bias += self.learning_rate * y_[idx]
                    updated = True
            if not updated:
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output <= 0, -1, 1)


def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features].copy()
    y = df['Survived'].copy()

    X['Age'] = X['Age'].fillna(X['Age'].mean())
    X['Fare'] = X['Fare'].fillna(X['Fare'].mean())

    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train.values, X_test.values, y_train.values, y_test.values


def train_and_evaluate(X_train, X_test, y_train, y_test, lr=0.01, max_iter=1000):
    perceptron = Perceptron(learning_rate=lr, max_iter=max_iter)
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)
    y_pred = np.where(y_pred == -1, 0, 1)
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    # 混淆矩阵格式：[[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    return accuracy, tn, fp, fn, tp  #


if __name__ == "__main__":
    file_path = "../Tantic/train.csv"
    X_train, X_test, y_train, y_test = preprocess_data(file_path)

    accuracy, tn, fp, fn, tp = train_and_evaluate(X_train, X_test, y_train, y_test, lr=0.01, max_iter=2000)

    print(f"感知机预测泰坦尼克号幸存者的准确率：{accuracy:.4f}")
    cm = np.array([[tn, fp], [fn, tp]])
    classes = ['Survived', 'Dead']
    plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix')

"""
0.7654(iter = 1000);0.7765(iter = 2000);0.7933(3000);0.7654(5000)
"""