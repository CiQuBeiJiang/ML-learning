# pandas数据清理

整理自知乎，版权归原作者所有。[作者：CodeCrafter](https://www.zhihu.com/question/486515583/answer/1963287734132121742)

## 第一阶段：望闻问切

> 搞清楚数据是什么，有什么类型，有什么问题

### 1. 加载数据：

用`pd.read_csv()`或者`pd.read_excel()`

### 2. 快速分析

- `df.head(5)`：看看前5行数据大概长什么样
- `df.info()`：会告诉你以下信息
  - 总共有多少行（RangeIndex）
  - 每一列的名字、非空值的数量、数据类型（Dtype）
  - `df.descrive()`：对所有数值类型的列，做一个统计概要，包括`count`、`mean`、`std`、`min`、`25%/50%/75%分位数`和`max`

## 第二阶段：定向打击

> 选取特定数据

- `.loc`：按标签（行名和列名）索引

  - 选单行：`df.loc[行索引名]`
  - 选多行：`df.loc[[行索引名1，行索引名2]]`
  - 选某几列：`df.loc[:,'列名1'，'列名2']`
  - 布尔索引

  ```python
  # 筛选出城市为'北京'的所有数据
  df.loc[df['city'] == '北京']
  
  # 筛选出年龄大于30且城市为'上海'的数据
  df.loc[(df['age'] > 30) & (df['city'] == '上海')]
  
  # 筛选出城市为'北京'或'深圳'的数据
  df.loc[df['city'].isin(['北京', '深圳'])]
  ```

- `.iloc`：按位置索引
  - 选前5行：`df.iloc[:5]`
  - 选第1、3、5行：`df.iloc[[0,2,4]]`
  - 选取第1行、第2列的值：`df.iloc[0,1]`

## 第三阶段：数据清洗与转换

### 1. 处理缺失值

`isnull()`、`fillna()`（补）、`dropna()`（删）

- `df.isnull().sum()`：快速统计每列有多少个缺失值
- `df.dropna()`：删除含有缺失值的列（某行数据大部分缺失，或某个特征列是空的时使用）
- `df.fillna()`：填充缺失值
  - 用前一个具体的值填充：`df['age'].fillna(0)`
  - 用均值或者中位数填充：`df['age'].fillna(df[''age].mean())`
  - 用前一个/后一个值填充：`df.fillna(method = 'ffill（向前填充）/bfill（向后填充）')`

### 2.数据类型转换

`astype()`

- 把价格列从字符串转为浮点数`df['price'] = df['price'].astype(float)`
- 处理日期`df['order_time'] = pd.to_datetime(df['order_time'])`

### 3.字符串处理

`.str`访问器

- 筛选出包含“手机”的行：`df['category'].str.contains('手机')`
- 按空格拆分agent字符串，并取第一个元素：`df['agent'].str.split(' ').str[0]`
- 把价格中的货币符号去掉然后转成数字`df['price_str'].str.replace("￥",'').astype(float)`

### 4. 创建新列/特征工程

```python
# 根据价格和数量计算总价
df['total_price'] = df['price'] * df['quantity']

# 从日期时间中提取年份和月份
df['order_year'] = df['order_time'].dt.year
df['order_month'] = df['order_time'].dt.month

# 根据年龄做一个分段
def age_group(age):
    if age < 18:
        return '少年'
    elif 18 <= age < 35:
        return '青年'
    else:
        return '中老年'

df['age_group'] = df['age'].apply(age_group)
```

- `apply()`对列中每个元素应用函数(尽量少用)
- `map()`映射：`df['gender'].map({'男': 1, '女': 0})`
- `applymap()`：作用域DataFrame里的每个元素

## 第四阶段：聚类分析

- `df.groupby('分组列')`
- 聚合函数：
  - `df.groupby('city')['salary'].mean()`：计算每个城市的平均薪资。`.count`、`.max()`、`.min()`均适用
  - `.agg()`：对不同的列使用不同的函数，如：

```python
# 对每个城市, 计算平均薪资和最大年龄
df.groupby('city').agg(
    avg_salary=('salary', 'mean'),
    max_age=('age', 'max'),
    user_count=('user_id', 'nunique') 
    # nunique 用于计算去重后的数量)
```

## 第五阶段：数据的整合

> 连接分散的多张表中的数据

- `pd.concat([df_1,df_2])`：“堆叠”，两个表完全相同时，可以合并成一个表
- `pd.merge()`：SQL里的`join`，可以根据一个或多个共同的键，横向链接两个表（`on`指定连接的键，`inner`、`left`、`right`、`outer`）

```python
# user_df: user_id, user_name, city
# order_df: order_id, user_id, price

# 把订单表和用户信息表连接起来
merged_df = pd.merge(order_df, user_df, on='user_id', how='left')
```

