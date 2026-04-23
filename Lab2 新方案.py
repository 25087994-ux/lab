import numpy as np
import pandas as pd

# 1.Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# 8。Load dataset
# 1. 读取数据
df = pd.read_csv(r"D:\python\新建文件夹\lab2\WorldEnergy.csv")
# 2. 先打印所有列名
print("="*30)
print("DataFrame 所有列名：")
print(df.columns.tolist())
print("="*30)
# 3. 再运行其他代码
print("\n前5行数据：")
print(df.head())# 1. 读取数据
df = pd.read_csv(r"D:\python\新建文件夹\lab2\WorldEnergy.csv")

# 9.Check structure and data types
df.info()
# Check for missing and duplicated records
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDuplicated rows:", df.duplicated().sum())

#10. Descriptive statistics for numeric columns
df.describe()
print("\nDescriptive statistics:\n", df.describe())

#11. Frequency count for categorical variable 'country'
print("\nDescriptive statistics:\n",df['country'].value_counts())

# 12.Correlation between numeric features
print("\nDescriptive statistics:\n",df.corr(numeric_only=True))

#3-6 Group-wise average power consumption per country and population
print("\nDescriptive statistics:\n",df.groupby('country')['population'].mean().sort_values(ascending=False))
print("\nDescriptive statistics:\n",df.groupby('country')['population'].mean())

#methodA
# 1. 计算gdp列的均值和标准差（变量名和gdp对应，更清晰）
mean_gdp = np.mean(df['gdp'])
std_gdp = np.std(df['gdp'])
# 2. 计算Z-score，新建一个列gdp_zscore存储，不覆盖原数据
df['gdp_zscore'] = (df['gdp'] - mean_gdp) / std_gdp
# 3. 筛选Z-score绝对值>3的异常值
df_anomalies_z = df[np.abs(df['gdp_zscore']) > 3]
# 4. 打印结果（脚本里必须用print！）
print("Anomalies detected (gdp Z-score method):", len(df_anomalies_z))
print("\nAnomaly data preview:")
print(df_anomalies_z.head())
print("输出methodA")

#methodB
Q1 = df['gdp'].quantile(0.25)
Q3 = df['gdp'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_anomalies_iqr = df[(df['gdp'] < lower_bound) | (df['gdp'] > upper_bound)]
print("Anomalies detected (IQR method):", len(df_anomalies_iqr))
print(df_anomalies_iqr.head())
print("输出methodB")

# 5.12 Count anomalies by machine
print("\n异常值按 machine_id 统计：")
print(df_anomalies_iqr['year'].value_counts())
# Average power during anomaly by machine status
print("\n异常值按 country 分组的 gdp 均值：")
print(df_anomalies_iqr.groupby('country')['gdp'].mean())
print("异常gdp")


# 6.14Unique values in 'country'
df['country'].unique()

# 6.15Fill missing values with mean power
df['gdp'] = df['gdp'].fillna(df['gdp'].mean())
# Drop duplicates
df.drop_duplicates(inplace=True)
print("After cleaning:")
print("Missing gdp values:", df['gdp'].isnull().sum())
print("Total rows remaining:", len(df))
print("输出gdp")

#7总结
summary = {
    "Total Records": len(df),
    "Missing Values (GDP)": df['gdp'].isnull().sum(),
    "Outliers (IQR method on GDP)": len(df_anomalies_iqr),
    "Mean GDP": round(df['gdp'].mean(), 2),
    "GDP-Population Correlation": round(df['gdp'].corr(df['population']), 3),
    "GDP-Wind Consumption Correlation": round(df['gdp'].corr(df['wind_consumption']), 3),
}
# print显示
print("\n=== 数据总结报告 ===")
print(pd.DataFrame(summary.items(), columns=["Metric", "Value"]))

#分析图表
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv(r"D:\python\新建文件夹\lab2\WorldEnergy.csv")
df.head()


# 1.先按国家聚合GDP，取前10名
gdp_by_country = df.groupby('country')['gdp'].sum().reset_index()
gdp_by_country = gdp_by_country.sort_values('gdp', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x='country', y='gdp', data=gdp_by_country, palette='viridis')
plt.title('Top 10 Countries by Total GDP')
plt.xlabel('Country')
plt.ylabel('Total GDP')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



#2. Lineplot: Total GDP in different years
sns.lineplot(x='year', y='gdp', data=df, errorbar=None)
plt.title('Total GDP in different years')
plt.show()

# Barplot: Top 10 Countries (GDP vs Population)
# 1. 同时聚合GDP和人口，按GDP排序取前10
gdp_by_country = df.groupby('country').agg(
    gdp=('gdp', 'sum'),
    population=('population', 'sum')
).reset_index()
gdp_by_country = gdp_by_country.sort_values('gdp', ascending=False).head(10)

# 2. 画GDP柱状图 + 人口次轴线图（双指标同时展示）
ax = sns.barplot(x='country', y='gdp', data=gdp_by_country, palette='muted', errorbar=None)
ax2 = ax.twinx()  # 次坐标轴显示人口
sns.lineplot(x='country', y='population', data=gdp_by_country, color='red', marker='o', ax=ax2)
plt.xticks(rotation=45, ha='right')  # 国家名旋转避免重叠
plt.title('GDP vs Population of Top 10 Countries by GDP')
plt.tight_layout()  # 防止标签被截断
plt.show()