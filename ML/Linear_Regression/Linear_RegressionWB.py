import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pandas.tseries.holiday import USFederalHolidayCalendar  # 美国节假日（根据你的数据所在地区选择）
# 如果是中国节假日，可用第三方库：pip install chinese-holiday （安装后导入ChineseHolidayCalendar）


# 字体设置
plt.rcParams["font.family"] = ["Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# 1. 查看数据集基本情况
def look_dataset(df):
    print("前5行数据：")
    print(df.head())
    print("\n数据类型和缺失值：")
    print(df.info())
    print("\n每列缺失值数量：")
    print(df.isnull().sum())


# 2. 合并天气和骑行数据
def transfer_time(wx, bx):
    # 转换日期格式
    wx['DATE'] = pd.to_datetime(wx['DATE'], format='%d %m %Y')  # 天气数据日期
    bx['Date'] = pd.to_datetime(bx['Date'], format='%m/%d/%Y %I:%M:%S %p')  # 骑行数据日期

    # 计算每天的总骑行量（因为骑行数据是每小时的）
    bicycle_daily = bx.resample('D', on='Date')['Fremont Bridge Total'].sum().reset_index()
    bicycle_daily.columns = ['DATE', 'Daily_Bicycle_Total']  # 重命名列，方便合并

    # 合并两个数据集（按日期匹配）
    mx = pd.merge(wx, bicycle_daily, on='DATE', how='inner')
    return mx

def add_holiday_feature(mx):
    # 1. 生成节假日日历（以美国联邦节假日为例，数据是西雅图的Fremont大桥，符合）
    cal = USFederalHolidayCalendar()
    # 2. 获取数据集中所有日期所在的年份范围
    start_year = mx['DATE'].dt.year.min()
    end_year = mx['DATE'].dt.year.max()
    # 3. 生成该范围内的所有节假日日期
    holidays = cal.holidays(start=f'{start_year}-01-01', end=f'{end_year}-12-31')
    # 4. 新增特征：是否为节假日（1=是，0=否）
    mx['is_holiday'] = mx['DATE'].isin(holidays).astype(int)
    return mx


# 3. 数据预处理（重点优化：增加特征+处理缺失值）
def processing(mx):
    # 3.1 EDA
    # 星期几（1=周一，7=周日）
    mx['weekday'] = mx['DATE'].dt.weekday + 1  # dt.weekday返回0-6（周一到周日），+1变成1-7
    # 是否周末（1=是，0=否）
    mx['is_weekend'] = mx['weekday'].apply(lambda x: 1 if x in [6, 7] else 0)
    # 月份（1-12月，季节可能影响骑行）
    mx['month'] = mx['DATE'].dt.month
    # 年份（骑行人数可能会随年份变化）
    mx['year'] = mx['DATE'].dt.year
    # 季节（区分为四个季节）
    mx['season'] = mx['month'].apply(lambda x:
        1 if x in [3, 4, 5] else  # 春季
        2 if x in [6, 7, 8] else  # 夏季
        3 if x in [9, 10, 11] else  # 秋季
        4)  # 冬季
    # 添加节假日
    add_holiday_feature(mx)

    # 添加特征交互项
    # 1. 温度 × 是否周末（捕捉不同工作日的温度影响差异）
    mx['TAVG_weekend'] = mx['TAVG'] * mx['is_weekend']

    # 2. 降雨量 × 季节（捕捉不同季节的降雨影响差异）
    mx['PRCP_season'] = mx['PRCP'] * mx['season']

    # 3. 风速 × 温度（大风在寒冷时影响更大）
    mx['WSF2_TAVG'] = mx['WSF2'] * mx['TAVG']

    # 温度 × 节假日
    mx['TAVG_holiday'] = mx['TAVG'] * mx['is_holiday']

    # 降雨量 × 节假日
    mx['PRCP_holiday'] = mx['PRCP'] * mx['is_holiday']

    # 3.2 处理缺失值
    # 温度缺失：用季节和月份对应的时间填充
    mx['TAVG'] = mx.groupby(['season', 'month'])['TAVG'].transform(
        lambda group: group.fillna(group.mean())
    )
    # 降雨量缺失：假设为0
    mx['PRCP'] = mx['PRCP'].fillna(0)
    # 风速缺失：用季节+月份中位数分组填充
    mx['WSF2'] = mx.groupby(['season', 'month'])['WSF2'].transform(
        lambda group: group.fillna(group.median())
    )

    # 3.3 删除无用的列
    mx = mx.drop(['WT13', 'WT14', 'WT16', 'WT18', 'WT22'], axis=1, errors='ignore')

    return mx


# 4. 简单分析特征
def analyze_features(mx):
    # 计算相关性
    corr = mx[['PRCP', 'TAVG', 'WSF2', 'is_weekend', 'month', 'Daily_Bicycle_Total','year']].corr()
    print("\n特征相关性：")
    print(corr['Daily_Bicycle_Total'])  # 只看和骑行量的相关性


    # 多特征与骑行量的关系
    features_to_check = ['PRCP', 'TAVG', 'WSF2', 'month']
    plt.figure(figsize=(15, 10))
    for i, feat in enumerate(features_to_check, 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(x=mx[feat], y=mx['Daily_Bicycle_Total'], alpha=0.5)
        plt.xlabel(feat)
        plt.ylabel('每日骑行量')
        plt.title(f'{feat}与骑行量的关系')
    plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(8, 5))
    # # 子图1：骑行量直方图（看分布是否偏态）
    # plt.subplot(1, 2, 1)
    # sns.histplot(mx['Daily_Bicycle_Total'], kde=True)
    # plt.xlabel('每日骑行量')
    # plt.title('骑行量分布')
    #
    # # 子图2：箱线图（识别异常值）
    # plt.subplot(1, 2, 2)
    # sns.boxplot(y=mx['Daily_Bicycle_Total'])
    # plt.ylabel('每日骑行量')
    # plt.title('骑行量箱线图（圆点为异常值）')
    # plt.tight_layout()
    # plt.show()
    #
    # # 计算异常值阈值（四分位法）
    # Q1 = mx['Daily_Bicycle_Total'].quantile(0.25)
    # Q3 = mx['Daily_Bicycle_Total'].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    # outliers = mx[(mx['Daily_Bicycle_Total'] < lower_bound) | (mx['Daily_Bicycle_Total'] > upper_bound)]
    # print(f"\n异常值数量：{len(outliers)}（占总数据的{len(outliers) / len(mx) * 100:.2f}%）")
    #
    # # 显示所有数据列
    # pd.set_option('display.max_columns', None)
    # # 打印包含所有列的异常值完整记录
    # print("\n异常值完整记录：")
    # print(outliers)

    # 无异常值

    # 多重共线性检测（计算方差膨胀因子VIF，VIF>10说明共线性严重）
    # 选择数值型特征
    vif_features = ['PRCP', 'TAVG', 'WSF2', 'is_weekend', 'month', 'season', 'is_holiday','year']
    X_vif = mx[vif_features].dropna()
    vif_data = pd.DataFrame()
    vif_data['特征'] = vif_features
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(len(vif_features))]
    print("\n方差膨胀因子（VIF）：")
    print(vif_data.sort_values('VIF', ascending=False))

# 5. 训练和评估模型（用优化后的特征）
def train_predict(mx):
    warnings.filterwarnings('ignore')  # 忽略警告

    # 选择更多特征
    feature_columns = ['PRCP', 'TAVG', 'WSF2', 'is_weekend', 'month',
                       'season','TAVG_weekend', 'PRCP_season', 'WSF2_TAVG',
                       'TAVG_holiday', 'PRCP_holiday','year']

    # 确保没有缺失值
    mx_clean = mx.dropna(subset=feature_columns + ['Daily_Bicycle_Total'])

    # 划分特征（X）和目标（y）
    X = mx_clean[feature_columns]  # 输入特征
    y = mx_clean['Daily_Bicycle_Total']  # 要预测的骑行量

    # 拆分训练集（80%）和测试集（20%）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # random_state确保结果可重复
    )

    # 特征标准化（岭回归必需）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # 测试集也要标准化

    # 训练岭回归，调整alpha参数
    model = RidgeCV(alphas=[1,2,3,4,5,6,7,8,9,10], cv=5)  # 5折交叉验证
    model.fit(X_train_scaled, y_train)
    print("最优alpha:", model.alpha_)
    # 预测测试集：必须用标准化后的X_test_scaled
    y_pred = model.predict(X_test_scaled)

    # 评估效果
    print("\n模型评估指标：")
    print(f"R²分数：{r2_score(y_test, y_pred):.4f}")
    print(f"平均误差：{mean_absolute_error(y_test, y_pred):.2f}")
    print(f"均方根误差：{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    # 画预测值vs真实值
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 理想线（预测=真实）
    plt.xlabel('真实骑行量')
    plt.ylabel('预测骑行量')
    plt.title('预测值 vs 真实值')
    plt.show()



# 主函数：按顺序执行
if __name__ == '__main__':
    # 加载数据
    df_w = pd.read_csv("../Dataset/Weather_Bicycle/Weather_Station_data.csv")  # 天气数据
    df_b = pd.read_csv("../Dataset/Weather_Bicycle/Fremont_Bridge_Bicycle_Counter.csv")  # 骑行数据

    # 步骤1：合并数据
    mx = transfer_time(df_w, df_b)

    # 步骤2：查看数据
    # look_dataset(mx)

    # 步骤3：预处理数据
    mx = processing(mx)

    # 步骤4：简单分析特征
    analyze_features(mx)

    # 步骤5：训练和评估模型
    train_predict(mx)