import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1.Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# 8。Load dataset
df = pd.read_csv(r"D:\python\新建文件夹\lab2\WorldEnergy.csv")
# 打印所有列名
print("="*30)
print("DataFrame 所有列名：")
print(df.columns.tolist())
print("="*30)
print("\n前5行数据：")
print(df.head())# 读取数据
df = pd.read_csv(r"D:\python\新建文件夹\lab2\WorldEnergy.csv")

plt.rcParams["font.family"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(r"D:\python\新建文件夹\lab2\WorldEnergy.csv")

# ======================  筛选中、韩、日数据 ======================
country_list = ["China", "South Korea", "Japan", "中国", "韩国", "日本"]
df_selected = df[df["country"].isin(country_list)].copy()

df_china = df_selected[df_selected["country"].str.contains("China|中国", na=False)].dropna(subset=["gdp", "coal_prod_change_twh"])
df_korea = df_selected[df_selected["country"].str.contains("South Korea|韩国", na=False)].dropna(subset=["gdp", "coal_prod_change_twh"])
df_japan = df_selected[df_selected["country"].str.contains("Japan|日本", na=False)].dropna(subset=["gdp", "coal_prod_change_twh"])
# 分别拆分三个国家的数据
df_cn = df_selected[df_selected["country"].str.contains("China|中国", na=False)]
df_kr = df_selected[df_selected["country"].str.contains("South Korea|韩国", na=False)]
df_jp = df_selected[df_selected["country"].str.contains("Japan|日本", na=False)]

# ======================  缺失值分析函数 ======================
def calculate_missing_rate(data, country_name):
    print(f"==================== {country_name} 缺失值分析 ====================")
    missing_count = data.isnull().sum()
    missing_percent = (missing_count / len(data)) * 100
    missing_df = pd.DataFrame({
        "缺失数量": missing_count,
        "缺失占比(%)": missing_percent.round(2)
    })
    print(missing_df)
    print("\n")

# ======================  GDP统计函数 ======================
def gdp_statistics(data, country_name):
    print(f"==================== {country_name} GDP 统计指标 ====================")
    if data.empty:
        print("无有效GDP数据")
    else:
        print(f"GDP 均值：\t{data['gdp'].mean():.2f}")
        print(f"GDP 最大值：\t{data['gdp'].max():.2f}")
        print(f"GDP 最小值：\t{data['gdp'].min():.2f}")
        print(f"GDP 标准差：\t{data['gdp'].std():.2f}")
    print("-"*60, "\n")

# ====================== coal_prod_change_twh 统计函数 ======================
def coal_statistics(data, country_name):
    print(f"==================== {country_name} coal_prod_change_twh 统计指标 ====================")
    print("字段含义：煤炭产量变化量（单位：TWh）")
    if data.empty:
        print("无有效煤炭产量变化数据")
    else:
        print(f"均值：\t{data['coal_prod_change_twh'].mean():.2f}")
        print(f"最大值：\t{data['coal_prod_change_twh'].max():.2f}")
        print(f"最小值：\t{data['coal_prod_change_twh'].min():.2f}")
        print(f"标准差：\t{data['coal_prod_change_twh'].std():.2f}")
    print("-"*60, "\n")

# 缺失值分析
calculate_missing_rate(df_china, "中国")
calculate_missing_rate(df_korea, "韩国")
calculate_missing_rate(df_japan, "日本")

#  GDP分国家统计
gdp_statistics(df_china, "中国")
gdp_statistics(df_korea, "韩国")
gdp_statistics(df_japan, "日本")


# ====================== GDP可视化图表 ======================
countries = ["中国", "韩国", "日本"]
gdp_mean = [df_china["gdp"].mean(), df_korea["gdp"].mean(), df_japan["gdp"].mean()]
gdp_max = [df_china["gdp"].max(), df_korea["gdp"].max(), df_japan["gdp"].max()]
gdp_min = [df_china["gdp"].min(), df_korea["gdp"].min(), df_japan["gdp"].min()]
gdp_std = [df_china["gdp"].std(), df_korea["gdp"].std(), df_japan["gdp"].std()]

# --- 图表1：三国GDP均值对比柱状图 ---
plt.figure(figsize=(10, 5))
plt.bar(countries, gdp_mean, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.5)
plt.title("中国、韩国、日本 GDP 均值对比", fontsize=14)
plt.ylabel("GDP 数值", fontsize=12)
for i, v in enumerate(gdp_mean):
    plt.text(i, v + max(gdp_mean)*0.01, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.show()

# --- 图表2：三国GDP 最大/最小/均值 对比图 ---
x = np.arange(len(countries))
width = 0.25
plt.figure(figsize=(12, 6))
plt.bar(x - width, gdp_max, width, label="GDP 最大值", color="#d62728")
plt.bar(x, gdp_mean, width, label="GDP 均值", color="#1f77b4")
plt.bar(x + width, gdp_min, width, label="GDP 最小值", color="#7f7f7f")
plt.xlabel("国家", fontsize=12)
plt.ylabel("GDP 数值", fontsize=12)
plt.title("三国GDP 最大值、均值、最小值对比", fontsize=14)
plt.xticks(x, countries)
plt.legend()
plt.tight_layout()
plt.show()

# --- 图表3：GDP分布箱线图 ---
plt.figure(figsize=(8, 5))
gdp_data = [df_china["gdp"], df_korea["gdp"], df_japan["gdp"]]
plt.boxplot(gdp_data, tick_labels=countries)
plt.title("三国 GDP 分布箱线图", fontsize=14)
plt.ylabel("GDP 数值", fontsize=12)
plt.tight_layout()
plt.show()

# . coal_prod_change_twh
coal_statistics(df_china, "中国")
coal_statistics(df_korea, "韩国")
coal_statistics(df_japan, "日本")
countries = ["中国", "韩国", "日本"]
# GDP数据
gdp_mean = [df_china["gdp"].mean(), df_korea["gdp"].mean(), df_japan["gdp"].mean()]


# 煤炭产量变化数据
coal_mean = [df_china["coal_prod_change_twh"].mean(), df_korea["coal_prod_change_twh"].mean(), df_japan["coal_prod_change_twh"].mean()]
coal_max = [df_china["coal_prod_change_twh"].max(), df_korea["coal_prod_change_twh"].max(), df_japan["coal_prod_change_twh"].max()]
coal_min = [df_china["coal_prod_change_twh"].min(), df_korea["coal_prod_change_twh"].min(), df_japan["coal_prod_change_twh"].min()]

# --- 图表1：均值对比 ---
plt.figure(figsize=(10, 5))
plt.bar(countries, coal_mean, color=["#ff9999", "#66b3ff", "#99ff99"], width=0.5)
plt.title("中国、韩国、日本 coal_prod_change_twh 均值对比", fontsize=14)
plt.ylabel("煤炭产量变化量（TWh）", fontsize=12)
for i, v in enumerate(coal_mean):
    plt.text(i, v+max(coal_mean)*0.01, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.show()

# --- 图表2：最大/均值/最小 对比 ---
x = np.arange(len(countries))
width = 0.25
plt.figure(figsize=(12, 6))
plt.bar(x-width, coal_max, width, label="最大值", color="#d62728")
plt.bar(x, coal_mean, width, label="均值", color="#ff7f0e")
plt.bar(x+width, coal_min, width, label="最小值", color="#1f77b4")
plt.xlabel("国家", fontsize=12)
plt.ylabel("煤炭产量变化量（TWh）", fontsize=12)
plt.title("三国 coal_prod_change_twh 极值与均值对比", fontsize=14)
plt.xticks(x, countries)
plt.legend()
plt.tight_layout()
plt.show()

# --- 图表3：分布箱线图 ---
plt.figure(figsize=(8, 5))
coal_data = [df_china["coal_prod_change_twh"], df_korea["coal_prod_change_twh"], df_japan["coal_prod_change_twh"]]
plt.boxplot(coal_data, tick_labels=countries)
plt.title("三国 coal_prod_change_twh 分布箱线图", fontsize=14)
plt.ylabel("煤炭产量变化量（TWh）", fontsize=12)
plt.tight_layout()
plt.show()

names = ["中国", "韩国", "日本"]
gdp_mean = [df_cn["gdp"].mean(), df_kr["gdp"].mean(), df_jp["gdp"].mean()]
coal_mean = [df_cn["coal_prod_change_twh"].mean(), df_kr["coal_prod_change_twh"].mean(), df_jp["coal_prod_change_twh"].mean()]
pop_mean = [df_cn["population"].mean(), df_kr["population"].mean(), df_jp["population"].mean()]

# --- 图表1：人口均值对比 ---
plt.figure(figsize=(10, 5))
plt.bar(names, pop_mean, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
plt.title("中国、韩国、日本 平均人口对比", fontsize=14)
plt.ylabel("人口数量", fontsize=12)
for i, v in enumerate(pop_mean):
    plt.text(i, v + max(pop_mean)*0.01, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.show()

# --- 图表2：人口分布箱线图 ---
plt.figure(figsize=(8, 5))
pop_data = [df_cn["population"].dropna(), df_kr["population"].dropna(), df_jp["population"].dropna()]
plt.boxplot(pop_data, tick_labels=names)
plt.title("三国人口数据分布箱线图", fontsize=14)
plt.ylabel("人口数量", fontsize=12)
plt.tight_layout()
plt.show()

