import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load dataset
df = pd.read_csv(r"C:\Users\Angel\Desktop\assignment2\sobar-72.csv")
print("="*30)
print("DataFrame 所有列名：")
print(df.columns.tolist())
print("="*30)
print("\n前5行数据：")
print(df.head())
df = pd.read_csv(r"C:\Users\Angel\Desktop\assignment2\sobar-72.csv")

def calculate_missing_rate(data, data_name):
    print(f"==================== {data_name} 缺失值分析 =========================")
    missing_count = data.isnull().sum()
    missing_percent = (missing_count / len(data)) * 100
    missing_df = pd.DataFrame({
        "缺失数量": missing_count,
        "缺失占比(%)": missing_percent.round(2)
    })
    print(missing_df)
    print("\n")
df = pd.read_csv(r"C:\Users\Angel\Desktop\assignment2\sobar-72.csv")
calculate_missing_rate(df, data_name="宫颈癌行为风险数据集")

# ----------------------  重复值处理 ----------------------
print("重复行数量：", df.duplicated().sum())
if df.duplicated().sum() > 0:
    df = df.drop_duplicates()
    print("删除重复值后形状：", df.shape)
 # ====================== 异常值检测与清洗 ======================
    print("\n=== Outliers Detection ===")
    feature_cols = df.columns.drop('ca_cervix').tolist()  # 自动获取所有特征列（除了目标变量ca_cervix）
    for col in feature_cols:  # 检测所有评分列的异常值（超出1-15范围的数）
        outlier_count = ((df[col] < 1) | (df[col] > 15)).sum()  # 异常值定义：小于1 或 大于15
        print(f"{col} out of range (1-15): {outlier_count}")
    print("ca_cervix invalid values (not 0/1):",  # 检测目标变量ca_cervix的异常值（只能是0或1）
          ((df["ca_cervix"] != 0) & (df["ca_cervix"] != 1)).sum())
 # ---------------------- 清洗异常值 ----------------------
    for col in feature_cols:  # 过滤所有评分列中超出1-15范围的行
        df = df[(df[col] >= 1) & (df[col] <= 15)]
    df = df[(df["ca_cervix"] == 0) | (df["ca_cervix"] == 1)]  # 过滤目标变量不是0或1的行
    print("Data shape after cleaning outliers:", df.shape)


print("特征列数值范围：")
print(df.describe().loc[['min', 'max']])
print("\n目标变量的所有取值：", df['ca_cervix'].unique())
total_outliers = 0
for col in df.columns.drop('ca_cervix'):
    outlier_count = ((df[col] < 1) | (df[col] > 15)).sum()
    total_outliers += outlier_count
print(f"\n超出1-15范围的异常值总数：{total_outliers}")

# 绘制热力图
df = pd.read_csv(r"C:\Users\Angel\Desktop\assignment2\sobar-72.csv")
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.show()

# ===================== EDA 描述性统计分析 =====================
print(" 数据集全变量描述性统计（均值/最大值/最小值/标准差）")
df = pd.read_csv(r"C:\Users\Angel\Desktop\assignment2\sobar-72.csv")
result = df.agg(['mean', 'min', 'max', 'std']).round(2)
print(result)




plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv(r"C:\Users\Angel\Desktop\assignment2\sobar-72.csv")
behavior_cols = [
    'behavior_sexualRisk', 'behavior_eating', 'behavior_personalHygine',
    'socialSupport_emotionality', 'socialSupport_appreciation', 'socialSupport_instrumental',
    'empowerment_knowledge', 'empowerment_abilities', 'empowerment_desires'
]

intention_cols = [
    'intention_aggregation', 'intention_commitment',
    'attitude_consistency', 'attitude_spontaneity',
    'motivation_strength', 'motivation_willingness'
]

perception_cols = [
    'norm_significantPerson', 'norm_fulfillment',
    'perception_vulnerability', 'perception_severity'
]

# ===================== 检查列是否存在 =====================
all_cols = set(df.columns)
for col in behavior_cols + intention_cols + perception_cols:
    if col not in all_cols:
        raise ValueError(f"列 '{col}' 不存在于数据集中，请检查列名拼写")

# ===================== 计算聚合得分（均值） =====================
df['behavior_score'] = df[behavior_cols].mean(axis=1)
df['intention_score'] = df[intention_cols].mean(axis=1)
df['perception_score'] = df[perception_cols].mean(axis=1)

# ===================== 验证分组合并是否成功 =====================
print("="*60)
print("生成的三个新列：")
print(" behavior_score   (行为类聚合得分)")
print(" intention_score  (意向态度类聚合得分)")
print(" perception_score (规范感知类聚合得分)")
print("="*60)

print("\n【1】行为类包含的列（共{}列）：".format(len(behavior_cols)))
print(behavior_cols)

print("\n【2】意向态度类包含的列（共{}列）：".format(len(intention_cols)))
print(intention_cols)

print("\n【3】规范感知类包含的列（共{}列）：".format(len(perception_cols)))
print(perception_cols)

print("\n" + "="*60)
print("聚合得分的描述性统计（均值/标准差/最小/最大）：")
print(df[['behavior_score', 'intention_score', 'perception_score']].describe().round(3))

print("\n数据前5行（含新得分）：")
print(df[['ca_cervix', 'behavior_score', 'intention_score', 'perception_score']].head())


# ===================== 绘制箱线图 =====================
target = 'ca_cervix'
# 箱线图1：行为类
plt.figure(figsize=(10, 5))
sns.boxplot(x=target, y='behavior_score', data=df, hue=target, palette='Set2', legend=False)
plt.title('行为类聚合得分 vs 宫颈癌患病状态', fontsize=14)
plt.xlabel('是否患病（0=未患病，1=患病）')
plt.ylabel('行为得分（均值）')
plt.tight_layout()
plt.show()

# 箱线图2：意向态度类
plt.figure(figsize=(10, 5))
sns.boxplot(x=target, y='intention_score', data=df, hue=target, palette='Set3', legend=False)
plt.title('意向态度类聚合得分 vs 宫颈癌患病状态', fontsize=14)
plt.xlabel('是否患病（0=未患病，1=患病）')
plt.ylabel('意向态度得分（均值）')
plt.tight_layout()
plt.show()

# 箱线图3：规范感知类
plt.figure(figsize=(10, 5))
sns.boxplot(x=target, y='perception_score', data=df, hue=target, palette='Set1', legend=False)
plt.title('规范感知类聚合得分 vs 宫颈癌患病状态', fontsize=14)
plt.xlabel('是否患病（0=未患病，1=患病）')
plt.ylabel('规范感知得分（均值）')
plt.tight_layout()
plt.show()



# ===================== KMeans 聚类分析 =====================
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

cluster_features = ['behavior_score', 'intention_score', 'perception_score']
X_cluster = df[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

k = 3
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42, algorithm='lloyd')
df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=cluster_features)
print("\n=== 聚类中心（每个簇的三类得分均值）===")
print(cluster_centers_df.round(3))

print("\n=== 各簇样本分布与患病率 ===")
cluster_summary = df.groupby('Cluster')['ca_cervix'].agg(['count', 'mean'])
cluster_summary.columns = ['样本数', '患病率']
print(cluster_summary.round(3))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))


pairs = [
    ('behavior_score', 'intention_score', '行为 vs 意向态度'),
    ('behavior_score', 'perception_score', '行为 vs 规范感知'),
    ('intention_score', 'perception_score', '意向态度 vs 规范感知')
]

for ax, (x_col, y_col, title) in zip(axes, pairs):
    for cluster_id in range(k):
        cluster_data = df[df['Cluster'] == cluster_id]
        ax.scatter(
            cluster_data[x_col], cluster_data[y_col],
            label=f'簇 {cluster_id}',
            s=40,
            alpha=0.6,
            edgecolors='k'
        )
    ax.scatter(
        cluster_centers_df[x_col], cluster_centers_df[y_col],
        s=200, c='red', marker='*', edgecolors='black',
        label='聚类中心'
    )
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()


# ===================== 逻辑回归分析=====================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import numpy as np

feature_cols = ['behavior_score', 'intention_score', 'perception_score']
X = df[feature_cols]
y = df['ca_cervix']

print("\n=== 类别分布（患病状态）===")
print(y.value_counts())
print(f"患病比例: {y.mean():.3f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

print("\n=== 逻辑回归系数 ===")
coef_df = pd.DataFrame({
    '特征': feature_cols,
    '系数': model.coef_[0],
    '优势比(Odds Ratio)': np.exp(model.coef_[0])
})
print(coef_df.round(4))
print(f"截距(Intercept): {model.intercept_[0]:.4f}")

y_prob = model.predict_proba(X_test)[:, 1]   # 患病概率
y_pred = (y_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n准确率(Accuracy): {accuracy:.3f}")

print("\n混淆矩阵(Confusion Matrix):")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 可视化混淆矩阵
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['未患病', '患病'], yticklabels=['未患病', '患病'])
plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title('混淆矩阵')
plt.tight_layout()
plt.show()

print("\n分类报告(Classification Report):")
print(classification_report(y_test, y_pred, target_names=['未患病', '患病']))

#ROC曲线和AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--', label='随机猜测')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('逻辑回归 ROC 曲线')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
corr_data = df[feature_cols + ['ca_cervix']].corr()
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('特征与目标变量相关性热力图')
plt.tight_layout()
plt.show()