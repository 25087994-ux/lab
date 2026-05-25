import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load dataset
df = pd.read_csv(r"C:\Users\Angel\Desktop\作业1\hour.csv")
print("="*30)
print("DataFrame 所有列名：")
print(df.columns.tolist())
print("="*30)
print("\n前5行数据：")
print(df.head())
df = pd.read_csv(r"C:\Users\Angel\Desktop\作业1\hour.csv")


def calculate_missing_rate(data, data_name):
    print(f"==================== {data_name} 缺失值分析 ====================")
    missing_count = data.isnull().sum()
    missing_percent = (missing_count / len(data)) * 100
    missing_df = pd.DataFrame({
        "缺失数量": missing_count,
        "缺失占比(%)": missing_percent.round(2)
    })
    print(missing_df)
    print("\n")

bike_data = pd.read_csv(r"C:\Users\Angel\Desktop\作业1\hour.csv")
calculate_missing_rate(bike_data, "共享单车数据集（hour.csv）")

# ----------------------  重复值处理 ----------------------
print("重复行数量：", df.duplicated().sum())
if df.duplicated().sum() > 0:
    df = df.drop_duplicates()
    print("删除重复值后形状：", df.shape)

# ---------------------- 删除冗余列 ----------------------
df = df.drop(columns=["instant"])
print("删除冗余列后：", df.shape)

# =====================  转换日期格式 =====================
df['dteday'] = pd.to_datetime(df['dteday'])
print("dteday 转换后数据类型：", df['dteday'].dtype)

# ===================== 分类变量转换 =====================
df["season"] = df["season"].astype(int).map({1:"Spring",2:"Summer",3:"Fall",4:"Winter"})
df["weathersit"] = df["weathersit"].astype(int).map({
    1:"Clear",
    2:"Foggy / Cloudy",
    3:"Light rain / Light snow",
    4:"Heavy rain / Heavy snow"
})
df["yr"] = df["yr"].astype(int).map({0:2011, 1:2012})
df["holiday"] = df["holiday"].astype(int).map({0:"Non-Holiday",1:"Holiday"})
df["workingday"] = df["workingday"].astype(int).map({0:"Non-Working Day",1:"Working Day"})

print("===== Categorical Variables Conversion (Top 10 Rows) =====")
print(df[["season", "weathersit", "yr", "holiday"]].head(10))

# ===================== 异常值检测与清洗 =====================
print("\n=== Outliers Detection ===")
print("Humidity out of range (0-1):", ((df["hum"] < 0) | (df["hum"] > 1)).sum())
print("Negative total rental count:", (df["cnt"] < 0).sum())
print("Negative casual users:", (df["casual"] < 0).sum())
print("Negative registered users:", (df["registered"] < 0).sum())

df = df[(df["hum"] >=0) & (df["hum"] <=1)]
df = df[df["cnt"] >=0]
print("Data shape after cleaning outliers:", df.shape)

# ===================== 还原归一化数据 =====================
df["real_temp"] = df["temp"] * 41
df["real_atemp"] = df["atemp"] * 50
df["real_humidity"] = df["hum"] * 100
df["real_windspeed"] = df["windspeed"] * 67


numeric_cols = df.select_dtypes(include=[np.number]).columns


corr_matrix = df[numeric_cols].corr()
print("=== 相关系数矩阵 ===")
print(corr_matrix)

# 绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

print("\n=== 数值变量统计摘要 ===")
print(df.describe())

# ===================== EDA: 季节与总租赁量 =====================
season_rental = df.groupby('season')['cnt'].agg(['sum', 'mean']).reset_index()
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
season_rental['season'] = pd.Categorical(season_rental['season'], categories=season_order, ordered=True)
season_rental = season_rental.sort_values('season')
print("\n===== Season Rental Summary =====")
print(season_rental)
# 柱状图：各季节总租赁数量
plt.figure(figsize=(10, 5))
sns.barplot(x='season', y='sum', hue='season', data=season_rental, palette='Set2', edgecolor='black', legend=False)
plt.title('Total Bike Rentals by Season', fontsize=14)
plt.xlabel('Season')
plt.ylabel('Total Rental Count')

max_sum = season_rental['sum'].max()
for index, value in enumerate(season_rental['sum']):
    plt.text(index, value + max_sum*0.02, f'{value:,}', ha='center', fontsize=11)

plt.tight_layout()
plt.show()

# 折线图：各季节总租赁数量
plt.figure(figsize=(10, 5))
sns.lineplot(x='season', y='sum', data=season_rental, marker='o', linewidth=3, markersize=10, color='darkblue')
plt.title('Total Bike Rentals Trend by Season', fontsize=14)
plt.xlabel('Season')
plt.ylabel('Total Rental Count')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =====================  季节箱线图 =====================
print("===== 绘制季节箱线图 =====")
plt.figure(figsize=(10, 5))
sns.boxplot(x='season', y='cnt', hue='season', data=df, palette='Set2', legend=False)
plt.title('Bike Rental Count by Season', fontsize=14)
plt.xlabel('Season')
plt.ylabel('Total Rental Count (cnt)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
sns.boxplot(x='weathersit', y='cnt', hue='weathersit', data=df, palette='Set3', legend=False)
plt.title('Bike Rental Count by Weather Situation', fontsize=14)
plt.xlabel('Weather Condition')
plt.ylabel('Total Rental Count (cnt)')
plt.tight_layout()
plt.show()

print("\n===== Average Rental Count by Weather =====")
print(df.groupby('weathersit')['cnt'].mean().round(2))

plt.figure(figsize=(12, 5))
sns.scatterplot(x='real_temp', y='cnt', data=df, alpha=0.3, color='orange')
sns.regplot(x='real_temp', y='cnt', data=df, scatter=False, color='red')
plt.title('Temperature vs Total Bike Rental Count', fontsize=14)
plt.xlabel('Real Temperature (°C)')
plt.ylabel('Total Rental Count (cnt)')
plt.tight_layout()
plt.show()

print("\n===== Correlation between Temperature and Rental Count =====")
print(f"Correlation: {df['real_temp'].corr(df['cnt']).round(3)}")



# ===================== 单因素ANOVA分析（季节/天气/温度 vs 总租赁量cnt） =====================
print("\n===== 单因素ANOVA：季节 vs 总租赁量 =====")
model_season = ols(formula='cnt ~ C(season)', data=df).fit()  # C(season)将季节设为分类变量
anova_season = sm.stats.anova_lm(model_season, typ=2)
print(anova_season)

print("\n===== 单因素ANOVA：天气 vs 总租赁量 =====")
model_weather = ols(formula='cnt ~ C(weathersit)', data=df).fit()
anova_weather = sm.stats.anova_lm(model_weather, typ=2)
print(anova_weather)

print("\n===== 单因素ANOVA：温度分组 vs 总租赁量 =====")
df['temp_group'] = pd.cut(df['real_temp'],
                          bins=[0, 15, 25, 40],
                          labels=['Low Temp', 'Mid Temp', 'High Temp'])
model_temp = ols(formula='cnt ~ C(temp_group)', data=df).fit()
anova_temp = sm.stats.anova_lm(model_temp, typ=2)
print(anova_temp)

# ===================== Tukey事后检验 =====================
print("\n===== Tukey事后检验：季节 =====")
df_season_clean = df.dropna(subset=['season'])
tukey_season = pairwise_tukeyhsd(endog=df_season_clean['cnt'],  # 因变量：总租赁量
                                groups=df_season_clean['season'],  # 分组变量：季节
                                alpha=0.05)
print(tukey_season)

print("\n===== Tukey事后检验：天气 =====")
df_weather_clean = df.dropna(subset=['weathersit'])
tukey_weather = pairwise_tukeyhsd(endog=df_weather_clean['cnt'],
                                  groups=df_weather_clean['weathersit'],
                                  alpha=0.05)
print(tukey_weather)

print("\n===== Tukey事后检验：温度分组 =====")
df_temp_clean = df.dropna(subset=['temp_group'])
tukey_temp = pairwise_tukeyhsd(endog=df_temp_clean['cnt'],
                              groups=df_temp_clean['temp_group'],
                              alpha=0.05)
print(tukey_temp)


# ===================== 1. 季节 vs 租赁量：正态性 & 方差齐性检验 =====================
print("\n===== 季节 正态性 & 方差齐性检验 =====")
print("\n===== Shapiro-Wilk 正态性检验（季节模型残差） =====")
model_season = ols(formula='cnt ~ C(season)', data=df).fit()
residuals_season = model_season.resid
shapiro_test_season = stats.shapiro(residuals_season)
print(f"统计量: {shapiro_test_season.statistic:.4f}, p值: {shapiro_test_season.pvalue:.4f}")
if shapiro_test_season.pvalue > 0.05:
    print("判断: p>0.05 → 残差符合正态分布，ANOVA假设成立")
else:
    print("判断: p<0.05 → 残差不符合正态分布，ANOVA假设不成立")

print("\n===== Levene 方差齐性检验（季节） =====")
spring_cnt = df[df['season']=='Spring']['cnt'].dropna()
summer_cnt = df[df['season']=='Summer']['cnt'].dropna()
fall_cnt = df[df['season']=='Fall']['cnt'].dropna()
winter_cnt = df[df['season']=='Winter']['cnt'].dropna()
levene_test_season = stats.levene(
    spring_cnt,
    summer_cnt,
    fall_cnt,
    winter_cnt
)
print(f"统计量: {levene_test_season.statistic:.4f}, p值: {levene_test_season.pvalue:.4f}")
if levene_test_season.pvalue > 0.05:
    print("判断: p>0.05 → 方差齐性，ANOVA假设成立")
else:
    print("判断: p<0.05 → 方差不齐，ANOVA假设不成立")

# ===================== 2. 天气 vs 租赁量：正态性 & 方差齐性检验 =====================
print("\n===== 天气 正态性 & 方差齐性检验 =====")
print("\n===== Shapiro-Wilk 正态性检验（天气模型残差） =====")
model_weather = ols(formula='cnt ~ C(weathersit)', data=df).fit()
residuals_weather = model_weather.resid
shapiro_test_weather = stats.shapiro(residuals_weather)
print(f"统计量: {shapiro_test_weather.statistic:.4f}, p值: {shapiro_test_weather.pvalue:.4f}")
if shapiro_test_weather.pvalue > 0.05:
    print("判断: p>0.05 → 残差符合正态分布，ANOVA假设成立")
else:
    print("判断: p<0.05 → 残差不符合正态分布，ANOVA假设不成立")

print("\n===== Levene 方差齐性检验（天气） =====")
clear_cnt = df[df['weathersit']=='Clear']['cnt'].dropna()
foggy_cnt = df[df['weathersit']=='Foggy / Cloudy']['cnt'].dropna()
light_rain_cnt = df[df['weathersit']=='Light rain / Light snow']['cnt'].dropna()
heavy_rain_cnt = df[df['weathersit']=='Heavy rain / Heavy snow']['cnt'].dropna()
levene_test_weather = stats.levene(
    clear_cnt,
    foggy_cnt,
    light_rain_cnt,
    heavy_rain_cnt
)
print(f"统计量: {levene_test_weather.statistic:.4f}, p值: {levene_test_weather.pvalue:.4f}")
if levene_test_weather.pvalue > 0.05:
    print("判断: p>0.05 → 方差齐性，ANOVA假设成立")
else:
    print("判断: p<0.05 → 方差不齐，ANOVA假设不成立")

print("\n===== 温度分组 正态性 & 方差齐性检验 =====")
print("\n===== Shapiro-Wilk 正态性检验（温度模型残差） =====")
model_temp = ols(formula='cnt ~ C(temp_group)', data=df).fit()
residuals_temp = model_temp.resid
shapiro_test_temp = stats.shapiro(residuals_temp)
print(f"统计量: {shapiro_test_temp.statistic:.4f}, p值: {shapiro_test_temp.pvalue:.4f}")
if shapiro_test_temp.pvalue > 0.05:
    print("判断: p>0.05 → 残差符合正态分布，ANOVA假设成立")
else:
    print("判断: p<0.05 → 残差不符合正态分布，ANOVA假设不成立")

print("\n===== Levene 方差齐性检验（温度分组） =====")
low_temp_cnt = df[df['temp_group'] == 'Low Temp']['cnt'].dropna()
mid_temp_cnt = df[df['temp_group'] == 'Mid Temp']['cnt'].dropna()
high_temp_cnt = df[df['temp_group'] == 'High Temp']['cnt'].dropna()

levene_test_temp = stats.levene(
    low_temp_cnt,
    mid_temp_cnt,
    high_temp_cnt
)
print(f"统计量: {levene_test_temp.statistic:.4f}, p值: {levene_test_temp.pvalue:.4f}")
if levene_test_temp.pvalue > 0.05:
    print("判断: p>0.05 → 方差齐性，ANOVA假设成立")
else:
    print("判断: p<0.05 → 方差不齐，ANOVA假设不成立")

