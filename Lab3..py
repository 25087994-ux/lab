import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# Load dataset
df = pd.read_csv(r"D:\python\新建文件夹\lab2\WorldEnergy.csv")
# Preview dataset
print(df.head())
# statistical info数据集基础信息
print(df.info())
#数值列描述性统计
print(df.describe())

# 筛选中国、韩国、日本数据
country_list = ["China", "South Korea", "Japan", "中国", "韩国", "日本"]
df_selected = df[df["country"].isin(country_list)].copy()
df_cn = df_selected[df_selected["country"].str.contains("China|中国", na=False)]
df_kr = df_selected[df_selected["country"].str.contains("South Korea|韩国", na=False)]
df_jp = df_selected[df_selected["country"].str.contains("Japan|日本", na=False)]

# 检查筛选结果
print("\n===== 三国数据行数检查 =====")
print(f"中国：{len(df_cn)} 行")
print(f"韩国：{len(df_kr)} 行")
print(f"日本：{len(df_jp)} 行")

#缺失值分析
def calculate_missing_rate(data, country_name):
    print(f"\n==================== {country_name} 缺失值分析 ====================")
    missing_count = data.isnull().sum()
    missing_percent = (missing_count / len(data)) * 100
    missing_df = pd.DataFrame({
        "缺失数量": missing_count,
        "缺失占比(%)": missing_percent.round(2)
    })
    print(missing_df)

# 执行缺失值分析
calculate_missing_rate(df_cn, "中国")
calculate_missing_rate(df_kr, "韩国")
calculate_missing_rate(df_jp, "日本")

# --- 图1：GDP 三国箱线图 ---
plt.figure(figsize=(10, 6))
sns.boxplot(x='country', y='gdp', data=df_selected)
plt.title('GDP Distribution by Country (China/Korea/Japan)')
plt.ylabel('GDP')
plt.show()

# --- 图2：人口 三国箱线图 ---
plt.figure(figsize=(10, 6))
sns.boxplot(x='country', y='population', data=df_selected)
plt.title('Population Distribution by Country (China/Korea/Japan)')
plt.ylabel('Population')
plt.show()

# --- 图3：煤炭产量变化量 三国箱线图 ---
plt.figure(figsize=(10, 6))
sns.boxplot(x='country', y='coal_prod_change_twh', data=df_selected)
plt.title('Coal Production Change by Country (China/Korea/Japan)')
plt.ylabel('Coal Production Change (TWh)')
plt.show()

# 单因素ANOVA
print("\n===== 单因素ANOVA：GDP =====")
model_gdp = ols('gdp ~ C(country)', data=df_selected).fit()
anova_gdp = sm.stats.anova_lm(model_gdp, typ=2)
print(anova_gdp)

print("\n===== 单因素ANOVA：人口 =====")
model_pop = ols('population ~ C(country)', data=df_selected).fit()
anova_pop = sm.stats.anova_lm(model_pop, typ=2)
print(anova_pop)

print("\n===== 单因素ANOVA：煤炭产量变化 =====")
model_coal = ols('coal_prod_change_twh ~ C(country)', data=df_selected).fit()
anova_coal = sm.stats.anova_lm(model_coal, typ=2)
print(anova_coal)

# Tukey事后检验
# GDP 部分
df_gdp_clean = df_selected.dropna(subset=['gdp'])
tukey_gdp = pairwise_tukeyhsd(df_gdp_clean['gdp'], df_gdp_clean['country'], alpha=0.05)
print("===== Tukey事后检验：GDP =====")
print(tukey_gdp)

# 人口 部分（已按统一流程修改）
df_pop_clean = df_selected.dropna(subset=['population'])  # 新增缺失值处理，和其他变量保持一致
tukey_pop = pairwise_tukeyhsd(df_pop_clean['population'], df_pop_clean['country'], alpha=0.05)
print("\n===== Tukey事后检验：人口 =====")
print(tukey_pop)

# 煤炭产量变化 部分
df_coal_clean = df_selected.dropna(subset=['coal_prod_change_twh'])
tukey_coal = pairwise_tukeyhsd(df_coal_clean['coal_prod_change_twh'], df_coal_clean['country'], alpha=0.05)
print("\n===== Tukey事后检验：煤炭产量变化 =====")
print(tukey_coal)

# ==========  GDP 正态性 & 方差齐性检验 ==========
print("\n===== Shapiro-Wilk 正态性检验（GDP模型残差） =====")
residuals_gdp = model_gdp.resid
shapiro_test_gdp = stats.shapiro(residuals_gdp)
print(f"统计量: {shapiro_test_gdp.statistic:.4f}, p值: {shapiro_test_gdp.pvalue:.4f}")
if shapiro_test_gdp.pvalue > 0.05:
    print("判断：p>0.05 → 残差符合正态分布，ANOVA假设成立")
else:
    print("判断：p<0.05 → 残差不符合正态分布，ANOVA假设不成立")

print("\n===== Levene 方差齐性检验（GDP） =====")
levene_test_gdp = stats.levene(
    df_cn['gdp'].dropna(),
    df_kr['gdp'].dropna(),
    df_jp['gdp'].dropna()
)
print(f"统计量: {levene_test_gdp.statistic:.4f}, p值: {levene_test_gdp.pvalue:.4f}")
if levene_test_gdp.pvalue > 0.05:
    print("判断：p>0.05 → 方差齐性，ANOVA假设成立")
else:
    print("判断：p<0.05 → 方差不齐，ANOVA假设不成立")


# ==========  人口 正态性 & 方差齐性检验 ==========
print("\n===== Shapiro-Wilk 正态性检验（人口模型残差） =====")
residuals_pop = model_pop.resid
shapiro_test_pop = stats.shapiro(residuals_pop)
print(f"统计量: {shapiro_test_pop.statistic:.4f}, p值: {shapiro_test_pop.pvalue:.4f}")
if shapiro_test_pop.pvalue > 0.05:
    print("判断：p>0.05 → 残差符合正态分布，ANOVA假设成立")
else:
    print("判断：p<0.05 → 残差不符合正态分布，ANOVA假设不成立")

print("\n===== Levene 方差齐性检验（人口） =====")
levene_test_pop = stats.levene(
    df_cn['population'].dropna(),
    df_kr['population'].dropna(),
    df_jp['population'].dropna()
)
print(f"统计量: {levene_test_pop.statistic:.4f}, p值: {levene_test_pop.pvalue:.4f}")
if levene_test_pop.pvalue > 0.05:
    print("判断：p>0.05 → 方差齐性，ANOVA假设成立")
else:
    print("判断：p<0.05 → 方差不齐，ANOVA假设不成立")


# ==========  煤炭产量变化 正态性 & 方差齐性检验 ==========
print("\n===== Shapiro-Wilk 正态性检验（煤炭产量变化模型残差） =====")
residuals_coal = model_coal.resid
shapiro_test_coal = stats.shapiro(residuals_coal)
print(f"统计量: {shapiro_test_coal.statistic:.4f}, p值: {shapiro_test_coal.pvalue:.4f}")
if shapiro_test_coal.pvalue > 0.05:
    print("判断：p>0.05 → 残差符合正态分布，ANOVA假设成立")
else:
    print("判断：p<0.05 → 残差不符合正态分布，ANOVA假设不成立")

print("\n===== Levene 方差齐性检验（煤炭产量变化） =====")
levene_test_coal = stats.levene(
    df_cn['coal_prod_change_twh'].dropna(),
    df_kr['coal_prod_change_twh'].dropna(),
    df_jp['coal_prod_change_twh'].dropna()
)
print(f"统计量: {levene_test_coal.statistic:.4f}, p值: {levene_test_coal.pvalue:.4f}")
if levene_test_coal.pvalue > 0.05:
    print("判断：p>0.05 → 方差齐性，ANOVA假设成立")
else:
    print("判断：p<0.05 → 方差不齐，ANOVA假设不成立")


