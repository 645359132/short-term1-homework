import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data import DATA  # 导入清洗后的数据集
from aver import city_yearly_avg  # 导入每个城市每年的平均值数据

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# -------------------------- 手动添加中国主要城市经纬度 --------------------------
# 格式：城市名称: {经度(lon), 纬度(lat)}，可根据实际数据中的城市补充或修改
CHINA_CITY_COORDS = {
    "Beijing": {"lon": 116.4074, "lat": 39.9042},       # 北京
    "Shanghai": {"lon": 121.4737, "lat": 31.2304},     # 上海
    "Guangzhou": {"lon": 113.2644, "lat": 23.1291},    # 广州
    "Shenzhen": {"lon": 114.0669, "lat": 22.5429},    # 深圳
    "Chengdu": {"lon": 104.0665, "lat": 30.5728},      # 成都
    "Chongqing": {"lon": 106.5504, "lat": 29.5647},    # 重庆
    "Wuhan": {"lon": 114.3055, "lat": 30.5928},       # 武汉
    "Xi'an": {"lon": 108.9481, "lat": 34.2632},       # 西安
    "Nanjing": {"lon": 118.7781, "lat": 32.0415},     # 南京
    "Hangzhou": {"lon": 120.1551, "lat": 30.2741},    # 杭州
    "Tianjin": {"lon": 117.2007, "lat": 39.0842},     # 天津
    "Suzhou": {"lon": 120.6196, "lat": 31.3072},      # 苏州
    "Wuxi": {"lon": 120.3017, "lat": 31.5789},       # 无锡
    "Changsha": {"lon": 112.9822, "lat": 28.1944},    # 长沙
    "Qingdao": {"lon": 120.3696, "lat": 36.0672},     # 青岛
    "Ningbo": {"lon": 121.5469, "lat": 29.8683},      # 宁波
    "Dalian": {"lon": 121.6146, "lat": 38.9140},      # 大连
    "Xiamen": {"lon": 118.0894, "lat": 24.4798},      # 厦门
    "Shenyang": {"lon": 123.4315, "lat": 41.8056},    # 沈阳
    "Harbin": {"lon": 126.6376, "lat": 45.7560},      # 哈尔滨
    " Zhengzhou": {"lon": 113.6654, "lat": 34.7579},  # 郑州
    "Kunming": {"lon": 102.7126, "lat": 25.0406},     # 昆明
    "Guiyang": {"lon": 106.7072, "lat": 26.5978},     # 贵阳
    "Nanning": {"lon": 108.3162, "lat": 22.8240},     # 南宁
    "Hefei": {"lon": 117.2830, "lat": 31.8617},      # 合肥
    "Changzhou": {"lon": 119.9729, "lat": 31.7897},   # 常州
    "Foshan": {"lon": 113.1227, "lat": 23.0287},      # 佛山
    "Dongguan": {"lon": 113.7478, "lat": 23.0464},    # 东莞
    "Zhuhai": {"lon": 113.5647, "lat": 22.2753},     # 珠海
    "Wuhu": {"lon": 118.3773, "lat": 31.3377}         # 芜湖
}

# -------------------------- 筛选数据集中的中国城市 --------------------------
# 1. 获取数据集中所有城市名称
dataset_cities = set(DATA["city_name"].unique())
print(f"数据集中共包含 {len(dataset_cities)} 个城市")

# 2. 筛选出同时存在于经纬度字典和数据集的中国城市
matched_cities = [city for city in dataset_cities if city in CHINA_CITY_COORDS]
print(f"匹配到的中国城市数量：{len(matched_cities)}")
if not matched_cities:
    print("警告：未匹配到任何中国城市，请检查城市名称是否一致（如中英文、空格差异）")
    exit()

# -------------------------- 准备聚类分析数据 --------------------------
# 数据结构：城市名称、经度、纬度、年均碳排放量
analysis_data = []
for city in matched_cities:
    # 获取该城市经纬度
    coords = CHINA_CITY_COORDS[city]
    # 获取该城市碳排放年均值（从aver.py的city_yearly_avg中计算）
    if city in city_yearly_avg:
        yearly_values = city_yearly_avg[city]
        # 计算所有年份的平均碳排放（若有多年份数据）
        avg_carbon = sum(yearly_values.values()) / len(yearly_values)
        analysis_data.append({
            "city_name": city,
            "lon": coords["lon"],
            "lat": coords["lat"],
            "avg_carbon": avg_carbon
        })

# 转换为DataFrame便于后续处理
df = pd.DataFrame(analysis_data)
print(f"\n最终用于聚类的城市数据：\n{df.head(10)}")  # 打印前10条数据预览

# -------------------------- 肘部法确定最佳聚类数量 --------------------------
# 提取聚类特征（经度、纬度、年均碳排放量）并标准化
features = df[["lon", "lat", "avg_carbon"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 计算不同k值的误差（惯性值）
inertias = []
k_range = range(1, 11)  # 测试k=1到k=10
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # n_init=10避免警告
    kmeans.fit(scaled_features)
    inertias.append(kmeans.inertia_)

# 绘制肘部图并保存
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
plt.xlabel("聚类数量 (k)", fontsize=12)
plt.ylabel("误差平方和（惯性值）", fontsize=12)
plt.title("肘部法确定最佳KMeans聚类数量", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.xticks(k_range)
plt.savefig("elbow_method.png", dpi=300, bbox_inches="tight")
plt.close()
print("\n肘部图已保存为 elbow_method.png，请根据图中'肘部'位置调整聚类数量")

# -------------------------- 执行KMeans聚类 --------------------------
# 可根据肘部图结果修改n_clusters（默认设为4，可调整）
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(scaled_features)
print(f"\nKMeans聚类完成，共将 {len(df)} 个中国城市分为 {n_clusters} 类")

# -------------------------- 聚类结果可视化 --------------------------
# 创建2个子图：地理分布聚类图 + 碳排放-纬度关系图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 1. 子图1：城市地理分布聚类（经度vs纬度）
scatter1 = ax1.scatter(
    df["lon"], df["lat"],
    c=df["cluster"],
    cmap="viridis",  # 颜色映射，可替换为"tab10"、"plasma"等
    s=80,           # 点的大小
    alpha=0.8,       # 透明度
    edgecolors="white",  # 点的边框颜色
    linewidths=0.5   # 点的边框宽度
)
# 添加城市名称标签（避免重叠，仅标注部分关键城市）
key_cities = ["Beijing", "Shanghai", "Guangzhou", "Chengdu", "Chongqing"]
for _, row in df.iterrows():
    if row["city_name"] in key_cities:
        ax1.annotate(
            row["city_name"],
            (row["lon"], row["lat"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
ax1.set_xlabel("经度", fontsize=12)
ax1.set_ylabel("纬度", fontsize=12)
ax1.set_title("中国城市聚类地理分布", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)
# 添加颜色条
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label("聚类类别", fontsize=12)

# 2. 子图2：年均碳排放量vs纬度（展示碳排放与地理位置的关系）
scatter2 = ax2.scatter(
    df["avg_carbon"], df["lat"],
    c=df["cluster"],
    cmap="viridis",
    s=80,
    alpha=0.8,
    edgecolors="white",
    linewidths=0.5
)
# 添加城市名称标签
for _, row in df.iterrows():
    if row["city_name"] in key_cities:
        ax2.annotate(
            row["city_name"],
            (row["avg_carbon"], row["lat"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
ax2.set_xlabel("年均碳排放量", fontsize=12)
ax2.set_ylabel("纬度", fontsize=12)
ax2.set_title("中国城市年均碳排放量与聚类关系", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3)
# 添加颜色条
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label("聚类类别", fontsize=12)

# 调整子图间距并保存
plt.tight_layout()
plt.savefig("china_city_clusters.png", dpi=300, bbox_inches="tight")
plt.close()
print("聚类可视化图已保存为 china_city_clusters.png")

# -------------------------- 聚类结果分析与保存 --------------------------
# 1. 统计每个聚类的核心特征
cluster_stats = df.groupby("cluster").agg({
    "city_name": "count",          # 每个聚类的城市数量
    "lon": "mean",                 # 每个聚类的平均经度
    "lat": "mean",                 # 每个聚类的平均纬度
    "avg_carbon": ["mean", "min", "max"]  # 每个聚类的碳排放统计（均值、最小值、最大值）
}).round(4)

# 重命名列名，便于阅读
cluster_stats.columns = ["城市数量", "平均经度", "平均纬度", "平均碳排放量", "最小碳排放量", "最大碳排放量"]
print("\n-------------------------- 聚类结果统计 --------------------------")
print(cluster_stats)

# 2. 查看每个聚类包含的具体城市
print("\n-------------------------- 各聚类城市列表 --------------------------")
for cluster in sorted(df["cluster"].unique()):
    cluster_cities = df[df["cluster"] == cluster]["city_name"].tolist()
    print(f"聚类 {cluster}（共{len(cluster_cities)}个城市）：{', '.join(cluster_cities)}")

# 3. 保存详细结果到CSV文件（便于后续分析）
df.to_csv("china_city_cluster_results.csv", index=False, encoding="utf-8-sig")
print("\n详细聚类结果已保存为 china_city_cluster_results.csv")