import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 设置中文显示
plt.rcParams["font.family"] = ["Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# 1. 数据加载与基础探索
def load_and_explore(file_path):
    """加载数据并进行基础探索"""
    df = pd.read_csv(file_path)
    print("数据集形状：", df.shape)
    print("\n前5行数据：")
    print(df.head())
    print("\n数据类型与缺失值概览：")
    print(df.info())

    # 查看缺失值情况
    missing_rate = df.isnull().sum() / len(df) * 100
    missing_features = missing_rate[missing_rate > 0].sort_values(ascending=False)
    print("\n缺失率>0的特征（%）：")
    print(missing_features.round(2))

    # 价格分布可视化
    plot_price_distribution(df)

    return df


def plot_price_distribution(df):
    """绘制销售价格分布图表"""
    # 直方图+核密度图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['SalePrice'], kde=True, color='blue', edgecolor='black')
    plt.title('销售价格分布', fontsize=14)
    plt.xlabel('销售价格（美元）', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.show()

    # 箱线图（异常值检测）
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df['SalePrice'], color='orange')
    plt.title('销售价格箱线图（异常值检测）', fontsize=14)
    plt.ylabel('销售价格（美元）', fontsize=12)
    plt.show()

    # 价格统计信息
    print("\n销售价格基本统计：")
    print(df['SalePrice'].describe().round(2))


# 2. 数据预处理
def preprocess_data(df):
    """数据预处理：异常值处理、缺失值填充、特征编码"""
    # 分离特征和目标变量
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    # 处理异常值
    X, y = handle_outliers(X, y)

    # 区分数值型和分类型特征
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"\n数值型特征数量：{len(numeric_cols)}，分类型特征数量：{len(categorical_cols)}")

    # 缺失值填充
    X = fill_missing_values(X, numeric_cols, categorical_cols)

    # 分类型特征编码
    X_encoded = encode_categorical_features(X, categorical_cols)

    # 特征筛选（去除高相关性冗余特征）
    X_selected = select_features(X_encoded, y)

    return X_selected, y


def handle_outliers(X, y):
    """处理异常值（基于箱线图和核心特征）"""
    # 合并特征和目标变量便于筛选
    data = pd.concat([X, y], axis=1)

    # 1. 基于销售价格的异常值（IQR法则）
    Q1 = data['SalePrice'].quantile(0.25)
    Q3 = data['SalePrice'].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data['SalePrice'] >= Q1 - 1.5 * IQR) & (data['SalePrice'] <= Q3 + 1.5 * IQR)]

    # 2. 基于地上面积的异常值（面积大但价格低）
    data = data[~((data['GrLivArea'] > 4000) & (data['SalePrice'] < 300000))]

    # 3. 地下室面积异常值（过大）
    data = data[data['TotalBsmtSF'] <= 6000]

    print(f"处理异常值后，样本数从{len(X)}减少到{len(data)}")
    return data.drop('SalePrice', axis=1), data['SalePrice']


def fill_missing_values(X, numeric_cols, categorical_cols):
    """填充缺失值：数值型用中位数，分类型用众数"""
    # 数值型特征填充
    numeric_imputer = SimpleImputer(strategy='median')
    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

    # 分类型特征填充
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])

    # 验证填充效果
    print(f"缺失值填充后，总缺失值数量：{X.isnull().sum().sum()}")
    return X


def encode_categorical_features(X, categorical_cols):
    """分类型特征编码：低基数用独热编码，高基数用标签编码"""
    # 划分低基数（≤10类）和高基数（>10类）特征
    low_card_cols = [col for col in categorical_cols if X[col].nunique() <= 10]
    high_card_cols = [col for col in categorical_cols if X[col].nunique() > 10]

    # 1. 低基数特征：独热编码
    if low_card_cols:
        onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        low_card_encoded = onehot_encoder.fit_transform(X[low_card_cols])
        low_card_df = pd.DataFrame(
            low_card_encoded,
            columns=onehot_encoder.get_feature_names_out(low_card_cols),
            index=X.index
        )
    else:
        low_card_df = pd.DataFrame(index=X.index)

    # 2. 高基数特征：标签编码
    high_card_df = X[high_card_cols].copy()
    for col in high_card_cols:
        le = LabelEncoder()
        high_card_df[col] = le.fit_transform(high_card_df[col])

    # 合并编码后的特征与数值型特征
    numeric_df = X.drop(categorical_cols, axis=1)
    X_encoded = pd.concat([numeric_df, low_card_df, high_card_df], axis=1)

    print(f"特征编码后，总特征数从{X.shape[1]}增加到{X_encoded.shape[1]}")
    return X_encoded


def select_features(X, y, threshold=0.8):
    """筛选特征：去除高相关性冗余特征"""
    # 计算特征间相关性
    corr_matrix = X.corr().abs()

    # 找出高相关性特征对
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # 剔除冗余特征（保留与目标变量相关性更高的）
    if to_drop:
        X_selected = X.drop(to_drop, axis=1)
        print(f"剔除{len(to_drop)}个高相关性冗余特征，剩余特征数：{X_selected.shape[1]}")
        return X_selected
    return X


# 3. 模型训练与评估
def train_and_evaluate(X, y):
    """划分数据集、训练模型并评估"""
    # 划分训练集、验证集、测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42  # 0.25*0.8=0.2
    )
    print(f"\n数据集划分：训练集{len(X_train)}，验证集{len(X_val)}，测试集{len(X_test)}")

    # 训练基准模型：线性回归
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    print("\n===== 线性回归模型 =====")
    evaluate_model(lr_model, X_val, y_val, "验证集")

    # 训练进阶模型：随机森林（带调参）
    best_rf_model = train_random_forest(X_train_val, y_train_val)
    print("\n===== 最优随机森林模型 =====")
    evaluate_model(best_rf_model, X_test, y_test, "测试集")

    # 可视化特征重要性和预测效果
    plot_feature_importance(best_rf_model, X_train)
    plot_prediction_vs_true(best_rf_model, X_test, y_test)

    return best_rf_model


def train_random_forest(X_train_val, y_train_val):
    """训练随机森林并通过网格搜索调参"""
    # 定义参数网格
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [8, 10, 12],
        'min_samples_split': [2, 5]
    }

    # 网格搜索
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train_val, y_train_val)

    print(f"随机森林最优参数：{grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_model(model, X, y, set_name):
    """评估模型性能（RMSE和R²）"""
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    print(f"{set_name} RMSE：{rmse:.2f} 美元")
    print(f"{set_name} R²分数：{r2:.4f}")
    return rmse, r2


def plot_feature_importance(model, X_train):
    """绘制特征重要性Top10"""
    importances = model.feature_importances_
    features = X_train.columns
    indices = np.argsort(importances)[-10:]  # 取Top10

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('特征重要性', fontsize=12)
    plt.title('随机森林特征重要性Top10', fontsize=14)
    plt.show()


def plot_prediction_vs_true(model, X_test, y_test):
    """绘制预测值vs真实值散点图"""
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 理想线
    plt.title('预测房价 vs 真实房价', fontsize=14)
    plt.xlabel('真实房价（美元）', fontsize=12)
    plt.ylabel('预测房价（美元）', fontsize=12)
    plt.show()


# 主函数
def main():
    file_path = "../Dataset/house-prices-advanced-regression-techniques/train.csv"

    # 1. 加载与探索数据
    df = load_and_explore(file_path)

    # 2. 数据预处理
    X_processed, y = preprocess_data(df)

    # 3. 模型训练与评估
    best_model = train_and_evaluate(X_processed, y)
    print("\n流程结束，最优模型已生成")


if __name__ == '__main__':
    main()