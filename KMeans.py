# 全球城市双维度KMeans聚类分析（经纬度+碳排放）
# 融合地理空间特征与碳排放特征，实现更精准的城市聚类

# 1. 导入必要库
from aver import city_yearly_avg as avg  # 从aver.py导入城市年均值字典
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score  # 聚类效果评估
import warnings

warnings.filterwarnings('ignore')  # 忽略无关警告

# 2. 全球城市经纬度字典（地理特征核心数据）
city_coords = {
    # 欧洲城市
    'Amsterdam': (52.3676, 4.9041),  # 阿姆斯特丹
    'Barcelona': (41.3874, 2.1686),  # 巴塞罗那
    'Berlin': (52.5200, 13.4050),  # 柏林
    'Copenhagen': (55.6761, 12.5683),  # 哥本哈根
    'Helsinki': (60.1699, 24.9384),  # 赫尔辛基
    'London': (51.5074, -0.1278),  # 伦敦
    'Lyon': (45.7640, 4.8357),  # 里昂
    'Madrid': (40.4168, -3.7038),  # 马德里
    'Marseille': (43.2965, 5.3698),  # 马赛
    'Milan': (45.4642, 9.1900),  # 米兰
    'Moscow': (55.7558, 37.6176),  # 莫斯科
    'Munich': (48.1351, 11.5820),  # 慕尼黑
    'Nice': (43.7102, 7.2620),  # 尼斯
    'Oslo': (59.9139, 10.7522),  # 奥斯陆
    'Paris': (48.8566, 2.3522),  # 巴黎
    'Rome': (41.9028, 12.4964),  # 罗马
    'Stockholm': (59.3293, 18.0686),  # 斯德哥尔摩
    # 亚洲城市
    'Bangkok': (13.7563, 100.5018),  # 曼谷
    'Beijing': (39.9042, 116.4074),  # 北京
    'Chongqing': (29.5647, 106.5504),  # 重庆
    'Chengdu': (30.5723, 104.0665),  # 成都
    'Hangzhou': (30.2741, 120.1551),  # 杭州
    'Jinan': (36.6754, 117.0219),  # 济南
    'Nanjing': (32.0603, 118.7969),  # 南京
    'New Delhi': (28.6139, 77.2090),  # 新德里
    'Osaka': (34.6937, 135.5023),  # 大阪
    'Qingdao': (36.0672, 120.3827),  # 青岛
    'Seoul': (37.5665, 126.9780),  # 首尔
    'Shanghai': (31.2304, 121.4737),  # 上海
    'Tokyo': (35.6762, 139.6503),  # 东京
    # 美洲城市
    'Bogota': (4.7110, -74.0721),  # 波哥大
    'Chicago': (41.8781, -87.6298),  # 芝加哥
    'Houston': (29.7604, -95.3698),  # 休斯顿
    'Los Angeles': (34.0522, -118.2437),  # 洛杉矶
    'Mexico City': (19.4326, -99.1332),  # 墨西哥城
    'Miami': (25.7617, -80.1918),  # 迈阿密
    'New York': (40.7128, -74.0060),  # 纽约
    'San Francisco': (37.7749, -122.4194),  # 旧金山
    'Santiago': (33.4489, -70.6693),  # 圣地亚哥
    'Seattle': (47.6062, -122.3321),  # 西雅图
    'Washington': (38.9072, -77.0369),  # 华盛顿
    # 大洋洲城市
    'Melbourne': (-37.8136, 144.9631),  # 墨尔本
    'Sydney': (-33.8688, 151.2093),  # 悉尼
    # 非洲及跨洲城市
    'Cape Town': (-33.9249, 18.4241),  # 开普敦
    'Istanbul': (41.0082, 28.9784)  # 伊斯坦布尔
}

# 3. 数据验证（确保双维度数据完整）
print("=" * 70)
print("全球城市双维度数据验证（经纬度+碳排放）")
print("=" * 70)
# 筛选同时具备经纬度和碳排放数据的城市
valid_cities = [city for city in avg.keys() if city in city_coords]
if not valid_cities:
    raise ValueError("⚠️ 无有效城市数据！请检查经纬度字典与碳排放数据的城市名匹配度")

# 打印验证结果
total_cities = len(avg.keys())
valid_count = len(valid_cities)
print(f"📊 数据源总城市数：{total_cities} 个")
print(f"✅ 有效双维度城市数：{valid_count} 个（同时有经纬度和碳排放数据）")
print(f"❌ 无效城市数：{total_cities - valid_count} 个（经纬度或碳排放数据缺失）")

# 4. 双维度特征构建（核心步骤：融合地理+排放特征）
print(f"\n" + "=" * 70)
print("双维度特征构建（地理特征2维 + 碳排放特征N维）")
print("=" * 70)
# 步骤1：提取碳排放特征（多时间维度）
all_years = set()
for city in valid_cities:
    all_years.update(avg[city].keys())
sorted_years = sorted(all_years)
emission_feature_dim = len(sorted_years)  # 碳排放特征维度（年份数量）
print(f"🌍 地理特征：2维（纬度、经度）")
print(f"🔋 碳排放特征：{emission_feature_dim}维（{sorted_years[0]}-{sorted_years[-1]}年每年均值）")
print(f"📐 双维度总特征数：{2 + emission_feature_dim} 维")

# 步骤2：构建双维度特征矩阵
cities = []  # 城市名称
double_dim_features = []  # 双维度特征矩阵（每行：[纬度, 经度, 2019年排放, 2020年排放, ...]）
avg_2019 = []  # 2019年排放均值（用于结果分析）
latitudes = []  # 纬度（用于结果追溯）
longitudes = []  # 经度（用于结果追溯）

for city in valid_cities:
    cities.append(city)
    # 地理特征：纬度、经度（标准化前先保留原始值）
    lat, lon = city_coords[city]
    latitudes.append(lat)
    longitudes.append(lon)
    # 碳排放特征：每年均值
    year_data = avg[city]
    emission_features = [year_data[year] for year in sorted_years]
    # 2019年排放均值（用于后续统计）
    avg_2019.append(year_data[2019])
    # 融合双维度特征：地理特征 + 碳排放特征
    combined_features = [lat, lon] + emission_features
    double_dim_features.append(combined_features)

# 5. 双维度数据标准化（关键：消除不同特征的量纲差异）
# 问题：地理特征（如纬度30-60）与碳排放特征（如10-100）量级差异大，必须分别标准化
# 方案：拆分特征→分别标准化→重新合并
# 拆分特征：地理特征（前2列）、碳排放特征（后N列）
geo_features = [feat[:2] for feat in double_dim_features]  # 地理特征（纬度、经度）
emission_features = [feat[2:] for feat in double_dim_features]  # 碳排放特征

# 分别标准化（避免互相干扰）
geo_scaler = StandardScaler()
geo_features_scaled = geo_scaler.fit_transform(geo_features)

emission_scaler = StandardScaler()
emission_features_scaled = emission_scaler.fit_transform(emission_features)

# 重新合并标准化后的双维度特征
double_dim_features_scaled = np.hstack([geo_features_scaled, emission_features_scaled])
print(f"\n✅ 双维度特征标准化完成：")
print(f"   - 地理特征标准化后：均值≈0，标准差≈1（消除纬度/经度量级差异）")
print(f"   - 碳排放特征标准化后：均值≈0，标准差≈1（消除不同城市排放量级差异）")

# 6. 双维度KMeans聚类（优化参数确保稳定性）
print(f"\n" + "=" * 70)
print("双维度KMeans聚类（5类，适配双维度特征）")
print("=" * 70)
n_clusters = 5  # 双维度聚类建议5类（比单维度多1类，体现空间+排放的细分度）
kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,  # 固定随机种子，结果可复现
    n_init=30,  # 增加初始中心尝试次数，提升稳定性
    max_iter=500  # 足够迭代次数，确保收敛
)
cluster_labels = kmeans.fit_predict(double_dim_features_scaled)

# 聚类效果评估（轮廓系数：越接近1越好，>0.3为合理）
sil_score = silhouette_score(double_dim_features_scaled, cluster_labels)
print(f"📈 聚类效果评估：轮廓系数 = {sil_score:.3f}")
if sil_score > 0.5:
    print(f"   → 聚类效果优秀（轮廓系数>0.5）")
elif sil_score > 0.3:
    print(f"   → 聚类效果合理（轮廓系数>0.3）")
else:
    print(f"   → 聚类效果一般（建议调整聚类数量n_clusters）")

# 7. 双维度聚类结果整理与深度统计
# 创建结果数据框（包含原始特征，便于追溯）
cluster_result = pd.DataFrame({
    "城市名称": cities,
    "纬度": latitudes,
    "经度": longitudes,
    "2019年碳排放均值": avg_2019,
    "聚类标签": cluster_labels,
    # 新增：各年份排放均值（便于分析排放趋势）
    **{f"{year}年排放均值": [emission_features[i][idx] for i in range(len(emission_features))]
       for idx, year in enumerate(sorted_years)}
})

# 按聚类分组统计（突出双维度特征）
print(f"\n=== 双维度聚类结果详细统计 ===")
for cluster_id in range(n_clusters):
    cluster_data = cluster_result[cluster_result["聚类标签"] == cluster_id]
    # 1. 地理特征统计（纬度/经度范围，反映空间聚集性）
    avg_lat = cluster_data["纬度"].mean()
    avg_lon = cluster_data["经度"].mean()
    lat_range = f"{cluster_data['纬度'].min():.2f} - {cluster_data['纬度'].max():.2f}"
    lon_range = f"{cluster_data['经度'].min():.2f} - {cluster_data['经度'].max():.2f}"

    # 2. 碳排放特征统计（2019年数据，反映排放水平）
    min_emission = cluster_data["2019年碳排放均值"].min()
    max_emission = cluster_data["2019年碳排放均值"].max()
    avg_emission = cluster_data["2019年碳排放均值"].mean()

    # 3. 空间-排放特征关联分析（例如“欧洲低排放”“东亚高排放”）
    # 简单区域判断（基于经度范围）
    if 0 <= avg_lon <= 30:
        region = "欧洲"
    elif 70 <= avg_lon <= 150:
        region = "亚洲"
    elif -130 <= avg_lon <= -60:
        region = "美洲"
    elif 110 <= avg_lon <= 180:
        region = "大洋洲"
    elif -20 <= avg_lon <= 40:
        region = "非洲/跨洲"
    else:
        region = "跨区域"

    # 打印聚类详情（突出双维度关联）
    print(f"\n🔹 聚类{cluster_id}（共 {len(cluster_data)} 个城市）：")
    print(f"   🌍 地理特征：平均纬度{avg_lat:.2f}，平均经度{avg_lon:.2f}，范围（纬度：{lat_range}，经度：{lon_range}）")
    print(f"   🔋 排放特征：2019年平均排放{avg_emission:.2f}（范围：{min_emission:.2f} - {max_emission:.2f}）")
    print(
        f"   📌 双维度标签：{region}{'高排放' if avg_emission > cluster_result['2019年碳排放均值'].mean() else '低排放'}集群")
    print(f"   🏙️  包含城市：{', '.join(cluster_data['城市名称'].tolist())}")

# 8. 结果导出（保存双维度聚类结果，支持后续分析）
output_path = "全球城市双维度聚类结果（经纬度+碳排放）.csv"
cluster_result.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n" + "=" * 70)
print(f"✅ 双维度聚类结果已保存至：{output_path}")
print(f"📋 结果包含：城市名称、经纬度、聚类标签、2019年排放均值、各年份原始排放数据")
print("=" * 70)