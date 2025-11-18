import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate  # 学习率
        self.max_iter = max_iter  # 最大迭代次数
        self.weights = None  # 权重
        self.bias = None  # 偏置

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 确保标签是1和-1
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.max_iter):
            updated = False
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) <= 0
                if condition:
                    # 更新规则
                    self.weights += self.learning_rate * y_[idx] * x_i
                    self.bias += self.learning_rate * y_[idx]
                    updated = True
            if not updated:
                break  # 没有误分类，提前结束

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output <= 0, -1, 1)