# 全球城市双维度KMeans聚类分析（经纬度+碳排放）
# 融合地理空间特征与碳排放特征，实现更精准的城市聚类

# 1. 导入必要库
import pandas as pd
import numpy as np
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score  # 聚类效果评估
from aver import calculate_average_by_year
from sklearn.preprocessing import MinMaxScaler


def kmeans_clustering():
    warnings.filterwarnings("ignore")  # 忽略无关警告
    avg = calculate_average_by_year()
    # 2. 全球城市经纬度字典（地理特征核心数据）
    city_coords = {
        "Anhui": (31.8617, 117.2853),  # 安徽（省会合肥坐标）
        "Beijing": (39.9042, 116.4074),  # 北京
        "Chongqing": (29.5630, 106.5516),  # 重庆
        "Fujian": (26.0789, 117.9874),  # 福建（省会福州坐标）
        "Gansu": (36.0611, 103.8343),  # 甘肃（省会兰州坐标）
        "Guangdong": (23.3790, 113.7633),  # 广东（省会广州坐标）
        "Guangxi": (22.8240, 108.3650),  # 广西（首府南宁坐标）
        "Guizhou": (26.5982, 106.7074),  # 贵州（省会贵阳坐标）
        "Hainan": (20.0174, 110.3492),  # 海南（省会海口坐标）
        "Hebei": (38.0428, 114.5149),  # 河北（省会石家庄坐标）
        "Heilongjiang": (45.8038, 126.5349),  # 黑龙江（省会哈尔滨坐标）
        "Henan": (34.7657, 113.7536),  # 河南（省会郑州坐标）
        "Hubei": (30.5950, 114.2999),  # 湖北（省会武汉坐标）
        "Hunan": (28.1127, 112.9838),  # 湖南（省会长沙坐标）
        "Inner Mongolia": (40.8175, 111.6708),  # 内蒙古（首府呼和浩特坐标）
        "Jiangsu": (32.0603, 118.7969),  # 江苏（省会南京坐标）
        "Jiangxi": (28.6765, 115.8922),  # 江西（省会南昌坐标）
        "Jilin": (43.8965, 125.3268),  # 吉林（省会长春坐标）
        "Liaoning": (41.7968, 123.4290),  # 辽宁（省会沈阳坐标）
        "Ningxia": (38.4680, 106.2732),  # 宁夏（首府银川坐标）
        "Qinghai": (36.6232, 101.7782),  # 青海（省会西宁坐标）
        "Shaanxi": (34.3416, 108.9398),  # 陕西（省会西安坐标）
        "Shandong": (36.6685, 116.9974),  # 山东（省会济南坐标）
        "Shanghai": (31.2304, 121.4737),  # 上海
        "Shanxi": (37.8735, 112.5624),  # 山西（省会太原坐标）
        "Sichuan": (30.5728, 104.0668),  # 四川（省会成都坐标）
        "Tianjin": (39.3434, 117.3616),  # 天津
        "Tibet": (29.6469, 91.1175),  # 西藏（首府拉萨坐标）
        "Xinjiang": (43.7930, 87.6277),  # 新疆（首府乌鲁木齐坐标）
        "Yunnan": (25.0438, 102.7097),  # 云南（省会昆明坐标）
        "Zhejiang": (30.2741, 120.1551),  # 浙江（省会杭州坐标）
    }

    # 3. 数据验证（确保与经纬度字典匹配）
    print("=" * 70)
    print("全球城市双维度数据验证（经纬度+碳排放）")
    print("=" * 70)

    # 提取经纬度字典中的城市名称集合
    coord_cities = set(city_coords.keys())
    # 提取碳排放数据中的城市名称集合
    emission_cities = set(avg.keys())

    # 筛选同时具备经纬度和碳排放数据的城市
    valid_cities = list(coord_cities.intersection(emission_cities))
    # 识别不匹配的城市（便于数据检查）
    missing_in_coord = list(emission_cities - coord_cities)  # 有排放数据但无经纬度
    missing_in_emission = list(coord_cities - emission_cities)  # 有经纬度但无排放数据

    if not valid_cities:
        raise ValueError(
            f"⚠️ 无有效城市数据！\n 经纬度字典缺失: {missing_in_coord}\n 排放数据缺失: {missing_in_emission}"
        )

    # 打印验证结果
    print(f" 经纬度字典包含城市数：{len(coord_cities)} 个")
    print(f" 碳排放数据包含城市数：{len(emission_cities)} 个")
    print(f" 有效双维度城市数：{len(valid_cities)} 个（同时存在于两者中）")
    print(f" 不匹配数据：")
    print(f"   - 有排放数据但无经纬度：{missing_in_coord}")
    print(f"   - 有经纬度但无排放数据：{missing_in_emission}")

    # 4. 双维度特征构建（基于经纬度字典）
    print(f"\n" + "=" * 70)
    print("双维度特征构建（地理特征2维 + 碳排放特征N维）")
    print("=" * 70)

    # 提取碳排放特征的时间维度
    all_years = set()
    for city in valid_cities:
        all_years.update(avg[city].keys())
    sorted_years = sorted(all_years)
    emission_feature_dim = len(sorted_years)

    print(f" 地理特征：2维（来自经纬度字典的纬度、经度）")
    print(
        f" 碳排放特征：{emission_feature_dim}维（{sorted_years[0]}-{sorted_years[-1]}年）"
    )
    print(f" 双维度总特征数：{2 + emission_feature_dim} 维")

    # 构建双维度特征矩阵
    cities = []  # 城市名称（与经纬度字典一致）
    double_dim_features = []  # 特征矩阵：[纬度, 经度, 各年排放数据]
    avg_2019 = []  # 2019年排放均值
    latitudes = []  # 原始纬度（来自经纬度字典）
    longitudes = []  # 原始经度（来自经纬度字典）

    for city in valid_cities:
        cities.append(city)
        # 从经纬度字典获取地理坐标
        lat, lon = city_coords[city]
        latitudes.append(lat)
        longitudes.append(lon)

        # 获取碳排放数据
        year_data = avg[city]
        emission_features = [year_data[year] for year in sorted_years]
        avg_2019.append(year_data.get(2019, np.nan))  # 处理可能的2019年数据缺失

        # 融合特征
        combined_features = [lat, lon] + emission_features
        double_dim_features.append(combined_features)

    # 过滤2019年数据缺失的样本
    valid_2019_mask = [not np.isnan(val) for val in avg_2019]
    if not all(valid_2019_mask):
        missing_2019 = [city for city, mask in zip(cities, valid_2019_mask) if not mask]
        print(f" 过滤{len(missing_2019)}个2019年数据缺失的城市：{missing_2019}")

        # 应用过滤
        cities = [c for c, m in zip(cities, valid_2019_mask) if m]
        latitudes = [l for l, m in zip(latitudes, valid_2019_mask) if m]
        longitudes = [lo for lo, m in zip(longitudes, valid_2019_mask) if m]
        avg_2019 = [a for a, m in zip(avg_2019, valid_2019_mask) if m]
        double_dim_features = [
            f for f, m in zip(double_dim_features, valid_2019_mask) if m
        ]

    # 5. 双维度数据标准化（适配经纬度特征）
    # 拆分特征：地理特征（来自经纬度字典）和碳排放特征
    geo_features = [feat[:2] for feat in double_dim_features]  # 纬度、经度
    emission_features = [feat[2:] for feat in double_dim_features]  # 排放数据

    # 分别标准化
    geo_scaler = StandardScaler()
    geo_features_scaled = geo_scaler.fit_transform(geo_features)

    emission_scaler = StandardScaler()
    emission_features_scaled = emission_scaler.fit_transform(emission_features)

    # 合并标准化后的特征
    double_dim_features_scaled = np.hstack(
        [geo_features_scaled, emission_features_scaled]
    )
    print(f"\n 双维度特征标准化完成：")
    print(f"   - 地理特征（经纬度）标准化后：均值≈0，标准差≈1")
    print(f"   - 碳排放特征标准化后：均值≈0，标准差≈1")

    # 6. 双维度KMeans聚类
    print(f"\n" + "=" * 70)
    print("双维度KMeans聚类（适配经纬度分布特征）")
    print("=" * 70)
    n_clusters = 5  # 适合经纬度+排放特征的聚类数量
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=30,  # 增加初始中心尝试次数，适配地理特征
        max_iter=500,
    )
    cluster_labels = kmeans.fit_predict(double_dim_features_scaled)

    # 聚类效果评估
    sil_score = silhouette_score(double_dim_features_scaled, cluster_labels)
    print(f" 聚类效果评估：轮廓系数 = {sil_score:.3f}")
    if sil_score > 0.4:
        print(f"   → 聚类效果优秀（双维度特征下）")
    elif sil_score > 0.2:
        print(f"   → 聚类效果合理（双维度特征下）")
    else:
        print(f"   → 建议调整聚类数量n_clusters")

    # 7. 聚类结果整理与统计
    # 创建结果数据框
    cluster_result = pd.DataFrame(
        {
            "城市名称": cities,
            "纬度（经纬度字典）": latitudes,
            "经度（经纬度字典）": longitudes,
            "2019年碳排放均值": avg_2019,
            "聚类标签": cluster_labels,
            # 添加各年份排放数据
            **{
                f"{year}年排放均值": [
                    emission_features[i][idx] for i in range(len(emission_features))
                ]
                for idx, year in enumerate(sorted_years)
            },
        }
    )

    # 按聚类分组统计
    print(f"\n=== 双维度聚类结果详细统计 ===")
    for cluster_id in range(n_clusters):
        cluster_data = cluster_result[cluster_result["聚类标签"] == cluster_id]
        if len(cluster_data) == 0:
            continue

        # 地理特征统计（基于经纬度字典数据）
        avg_lat = cluster_data["纬度（经纬度字典）"].mean()
        avg_lon = cluster_data["经度（经纬度字典）"].mean()
        lat_range = f"{cluster_data['纬度（经纬度字典）'].min():.2f} - {cluster_data['纬度（经纬度字典）'].max():.2f}"
        lon_range = f"{cluster_data['经度（经纬度字典）'].min():.2f} - {cluster_data['经度（经纬度字典）'].max():.2f}"

        # 碳排放特征统计
        min_emission = cluster_data["2019年碳排放均值"].min()
        max_emission = cluster_data["2019年碳排放均值"].max()
        avg_emission = cluster_data["2019年碳排放均值"].mean()

        # 区域判断（基于经纬度）
        if 70 <= avg_lon <= 150:  # 亚洲经度范围
            region = "亚洲"
        elif -130 <= avg_lon <= -60:  # 美洲经度范围
            region = "美洲"
        elif -20 <= avg_lon <= 40:  # 非洲/欧洲经度范围
            region = "非洲/欧洲"
        elif 110 <= avg_lon <= 180:  # 大洋洲经度范围
            region = "大洋洲"
        else:
            region = "跨区域"

        # 打印聚类详情
        print(f"\n 聚类{cluster_id}（共 {len(cluster_data)} 个城市）：")
        print(
            f"    地理特征：平均纬度{avg_lat:.2f}，平均经度{avg_lon:.2f}，范围（纬度：{lat_range}，经度：{lon_range}）"
        )
        print(
            f"    排放特征：2019年平均排放{avg_emission:.2f}（范围：{min_emission:.2f} - {max_emission:.2f}）"
        )
        print(
            f"    双维度标签：{region}{'高排放' if avg_emission > cluster_result['2019年碳排放均值'].mean() else '低排放'}集群"
        )
        print(f"    包含城市：{', '.join(cluster_data['城市名称'].tolist())}")

    # 8. 结果导出
    output_path = "全球城市双维度聚类结果（经纬度+碳排放）.csv"
    draw_map(cluster_result)
    cluster_result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n" + "=" * 70)
    print(f" 双维度聚类结果已保存至：{output_path}")
    print(f" 结果包含：城市名称、经纬度（来自经纬度字典）、聚类标签、各年份排放数据")
    print("=" * 70)


def draw_map(data):
    # ==================== 1. 准备工作：配置环境和中文字体 ====================
    # 配置matplotlib以支持中文显示
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用黑体
        plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
        print("中文字体 'SimHei' 设置成功。")
    except:
        print("警告：未找到 'SimHei' 字体。图表中的中文可能无法正常显示。")
        print(
            "请根据您的操作系统安装并配置中文字体（如 'Microsoft YaHei', 'PingFang SC' 等）。"
        )

    # ==================== 2. 模拟您的聚类结果数据 ====================
    cluster_result = pd.DataFrame(data)
    print("\n模拟的聚类结果数据：")
    print(cluster_result)

    # ==================== 3. 加载地图数据并进行数据转换 ====================
    map_file_path = "china-map.json"
    try:
        china_map = gpd.read_file(map_file_path)
        print(f"\n成功加载地图文件: '{map_file_path}'")
    except Exception as e:
        print(f"\n错误：无法加载地图文件 '{map_file_path}'。")
        print(
            f"请从 http://datav.aliyun.com/portal/school/atlas/area_selector 下载并放置到正确位置。"
        )
        print(f"详细错误信息: {e}")
        exit()

    # 将您的城市数据 DataFrame 转换为 GeoDataFrame
    # **注意这里的列名已根据您的描述更新**
    gdf_cities = gpd.GeoDataFrame(
        cluster_result,
        geometry=gpd.points_from_xy(
            cluster_result["经度（经纬度字典）"], cluster_result["纬度（经纬度字典）"]
        ),
    )

    # ==================== 4. 开始绘图 ====================
    for year in range(2019, 2026):
        fig, ax = plt.subplots(1, 1, figsize=(18, 15))

        # 1. 绘制底图
        china_map.plot(ax=ax, edgecolor="black", facecolor="whitesmoke", linewidth=0.5)

        # 2. 定义颜色和大小
        # 颜色逻辑保持不变
        colors = plt.cm.tab10.colors
        unique_labels = sorted(gdf_cities["聚类标签"].unique())
        cluster_colors = {
            label: colors[i % len(colors)] for i, label in enumerate(unique_labels)
        }
        point_colors = gdf_cities["聚类标签"].map(cluster_colors)

        # *** 关键修改：使用MinMaxScaler对碳排放数据进行归一化以确定大小 ***
        emission_col = f"{year}年排放均值"
        min_size, max_size = 50, 1000  # 定义气泡的最小和最大尺寸

        # 创建一个MinMaxScaler实例
        scaler = MinMaxScaler(feature_range=(min_size, max_size))

        # 对排放数据进行拟合和转换，注意要将Series转换为[[v1], [v2], ...]的格式
        emission_values = gdf_cities[emission_col].values.reshape(-1, 1)
        point_sizes = scaler.fit_transform(emission_values)

        # 3. 在地图上绘制气泡
        ax.scatter(
            gdf_cities.geometry.x,
            gdf_cities.geometry.y,
            s=point_sizes,
            c=point_colors,
            alpha=0.75,
            edgecolor="black",  # 添加边框让气泡更清晰
            linewidth=0.5,
            zorder=10,
        )

        # 4. 添加城市名称标签
        for x, y, label in zip(
            gdf_cities.geometry.x, gdf_cities.geometry.y, gdf_cities["城市名称"]
        ):
            ax.text(x + 0.5, y + 0.5, label, fontsize=10, ha="left")

        # 5. 美化图表和添加图例
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        ax.set_title(
            f"中国主要城市双维度聚类地理分布图 (基于{year}年数据)",
            fontdict={"fontsize": 22, "fontweight": "bold"},
        )
        ax.set_facecolor("#aed6f1")  # 使用一个柔和的蓝色作为背景

        # --- 创建自定义图例 ---
        # 聚类标签图例（颜色）
        legend_elements_cluster = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"聚类 {key}",
                markerfacecolor=value,
                markersize=12,
            )
            for key, value in sorted(cluster_colors.items())
        ]

        # *** 关键修改：创建更智能的大小图例 ***
        # 碳排放量图例（大小）
        min_e, max_e = gdf_cities[emission_col].min(), gdf_cities[emission_col].max()
        # 选择3个代表性的排放值：最小值，中间值，最大值
        legend_emission_values = [min_e, (min_e + max_e) / 2, max_e]

        # 使用之前创建的scaler来转换这些代表值，以获得正确的图例标记大小
        legend_marker_sizes = scaler.transform(
            pd.Series(legend_emission_values).values.reshape(-1, 1)
        )

        legend_elements_size = []
        for i, emission_val in enumerate(legend_emission_values):
            legend_elements_size.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    # **关键修改：使用.2f格式化浮点数，显示两位小数**
                    label=f"{emission_val:.2f}",
                    markerfacecolor="gray",
                    markersize=legend_marker_sizes[i] ** 0.5,
                )  # marker size在legend中与s的单位不同，通常需要开根号调整
            )

        # 合并图例并显示
        legend1 = ax.legend(
            handles=legend_elements_cluster,
            title="聚类类别",
            loc="lower left",
            fontsize=12,
            title_fontsize=14,
        )
        ax.add_artist(legend1)
        legend2 = ax.legend(
            handles=legend_elements_size,
            title=f"{year}年碳排放均值",
            loc="lower right",
            fontsize=12,
            title_fontsize=14,
        )

        # ==================== 6. 显示并保存图像 ====================
        plt.tight_layout()
        output_path = f"中国城市双维度聚类地理分布图-{year}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n可视化图表已成功保存至: '{output_path}'")
        plt.show()
