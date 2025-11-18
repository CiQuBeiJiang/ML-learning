import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# 1. 通用数据加载（支持CSV文件）
def load_data(file_path, target_column):
    """
    加载CSV格式的数据集
    :param file_path: CSV文件路径
    :param target_column: 标签列的名称（目标变量）
    :return: 特征数据X和标签数据y
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 分离特征和标签
    X = df.drop(columns=[target_column]).values  # 特征（排除标签列）
    y = df[target_column].values  # 标签（目标变量）

    print(f"数据加载完成：共{X.shape[0]}个样本，{X.shape[1]}个特征")
    return X, y


# 2. 配置数据集信息（这里替换成你的文件路径和标签列名）
file_path = "your_dataset.csv"  # 你的CSV文件路径
target_column = "label"  # 你的标签列名称（根据实际数据修改）

# 加载数据
X, y = load_data(file_path, target_column)

# 3. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 定义参数网格
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'degree': [2, 3, 4]  # 仅对poly核有效
}

# 6. 网格搜索
grid_search = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# 7. 训练与调优
grid_search.fit(X_train_scaled, y_train)

# 8. 输出结果
print("\n最佳参数组合:", grid_search.best_params_)
print("最佳交叉验证准确率:", grid_search.best_score_.round(4))

# 9. 测试集评估
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)
print("测试集准确率:", accuracy_score(y_test, y_pred).round(4))
