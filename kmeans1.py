# ==============================================================================
# 全球城市双维度KMeans聚类分析【终极合并版】
#
# 本脚本融合了两个版本的优点：
# 1. 采用多维度评估指标（轮廓系数、CH、DBI），对聚类模型进行更科学的评估。
# 2. 采用Geopandas进行专业级地理可视化，生成信息丰富的气泡地图。
# 3. 实现了评估指标和地理分布的双重可视化。
# 4. 保持了清晰的函数化代码结构，易于维护和扩展。
# ==============================================================================

# 1. 导入必要库
import pandas as pd
import numpy as np
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    silhouette_score,  # 轮廓系数
    calinski_harabasz_score,  # 卡林斯基-哈拉巴斯指数
    davies_bouldin_score,  # 戴维斯-布尔丁指数
)

# 假设aver.py中的数据加载逻辑与原始代码一致
from aver import calculate_average_by_year, CHINA_DATA


def draw_evaluation_plots(sil_score, ch_score, db_score):
    """
    可视化聚类评估的三个核心指标。
    这是来自版本B的优点，用于展示模型性能。
    """
    print("\n" + "=" * 70)
    print("正在生成聚类评估指标可视化图...")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("聚类模型性能评估指标", fontsize=16, fontweight="bold")

    # 1. 轮廓系数
    axes[0].bar(["轮廓系数"], [sil_score], color="cornflowerblue")
    axes[0].set_title("轮廓系数 (越高越好)")
    axes[0].set_ylabel("分数")
    axes[0].set_ylim(0, 1)
    axes[0].text(
        0, sil_score, f"{sil_score:.3f}", ha="center", va="bottom", fontsize=12
    )

    # 2. 卡林斯基-哈拉巴斯指数
    axes[1].bar(["CH 指数"], [ch_score], color="mediumseagreen")
    axes[1].set_title("卡林斯基-哈拉巴斯指数 (越高越好)")
    axes[1].text(0, ch_score, f"{ch_score:.1f}", ha="center", va="bottom", fontsize=12)

    # 3. 戴维斯-布尔丁指数
    axes[2].bar(["DB 指数"], [db_score], color="salmon")
    axes[2].set_title("戴维斯-布尔丁指数 (越低越好)")
    axes[2].text(0, db_score, f"{db_score:.3f}", ha="center", va="bottom", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = "聚类评估指标图.png"
    plt.savefig(output_path, dpi=300)
    print(f"聚类评估指标图已保存至: '{output_path}'")
    plt.show()


def draw_geographic_map(data):
    """
    在真实地图上绘制聚类结果的地理气泡图。
    这是来自版本A的优点，提供专业级的地理可视化。
    """
    print("\n" + "=" * 70)
    print("正在生成地理分布可视化图...")
    print("=" * 70)

    # 准备工作：配置中文字体
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
    except:
        print("警告：'SimHei' 字体未找到，中文可能无法正常显示。")

    # 加载地图数据
    map_file_path = "china-map.json"
    try:
        china_map = gpd.read_file(map_file_path)
    except Exception as e:
        print(f"错误：无法加载地图文件 '{map_file_path}'。请确保文件存在。")
        print(f"详细错误: {e}")
        return

    # 将DataFrame转换为GeoDataFrame
    gdf_cities = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(
            data["经度（经纬度字典）"], data["纬度（经纬度字典）"]
        ),
    )

    # 为2019到2025年的每一年生成一张图
    for year in range(2019, 2026):
        emission_col = f"{year}年排放均值"
        if emission_col not in gdf_cities.columns:
            print(f"警告：数据中缺少 '{emission_col}'，跳过该年份的地图绘制。")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(18, 15))
        china_map.plot(ax=ax, edgecolor="black", facecolor="whitesmoke", linewidth=0.5)

        # 定义颜色
        colors = plt.cm.tab10.colors
        unique_labels = sorted(gdf_cities["聚类标签"].unique())
        cluster_colors = {
            label: colors[i % len(colors)] for i, label in enumerate(unique_labels)
        }
        point_colors = gdf_cities["聚类标签"].map(cluster_colors)

        # 定义大小 (使用MinMaxScaler)
        min_size, max_size = 50, 1000
        scaler = MinMaxScaler(feature_range=(min_size, max_size))
        emission_values = gdf_cities[emission_col].values.reshape(-1, 1)
        point_sizes = scaler.fit_transform(emission_values)

        # 绘制气泡
        ax.scatter(
            gdf_cities.geometry.x,
            gdf_cities.geometry.y,
            s=point_sizes,
            c=point_colors,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
            zorder=10,
        )

        # 添加城市标签
        for x, y, label in zip(
            gdf_cities.geometry.x, gdf_cities.geometry.y, gdf_cities["城市名称"]
        ):
            ax.text(x + 0.5, y + 0.5, label, fontsize=10, ha="left")

        # 美化与图例
        ax.set_axis_off()
        ax.set_title(
            f"中国主要城市双维度聚类地理分布图 ({year}年)",
            fontdict={"fontsize": 22, "fontweight": "bold"},
        )
        ax.set_facecolor("#aed6f1")

        # 聚类图例
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

        # 大小图例
        min_e, max_e = gdf_cities[emission_col].min(), gdf_cities[emission_col].max()
        legend_emission_values = [min_e, (min_e + max_e) / 2, max_e]
        legend_marker_sizes = scaler.transform(
            np.array(legend_emission_values).reshape(-1, 1)
        )
        legend_elements_size = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{val:.2f}",
                markerfacecolor="gray",
                markersize=size**0.5,
            )
            for val, size in zip(legend_emission_values, legend_marker_sizes)
        ]

        legend1 = ax.legend(
            handles=legend_elements_cluster,
            title="聚类类别",
            loc="lower left",
            fontsize=12,
        )
        ax.add_artist(legend1)
        legend2 = ax.legend(
            handles=legend_elements_size,
            title=f"{year}年碳排放均值",
            loc="lower right",
            fontsize=12,
        )

        # 保存与显示
        plt.tight_layout()
        output_path = f"中国城市双维度聚类地理分布图-{year}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"可视化图表已成功保存至: '{output_path}'")
        plt.show()


def run_full_analysis():
    """主分析函数，整合数据处理、聚类、评估和可视化"""
    warnings.filterwarnings("ignore")

    # 准备工作：配置中文字体
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
    except:
        print("警告：'SimHei' 字体未找到，中文可能无法正常显示。")

    # 2. 数据加载与准备
    avg = calculate_average_by_year(CHINA_DATA)
    city_coords = {
        "Anhui": (31.8617, 117.2853),
        "Beijing": (39.9042, 116.4074),
        "Chongqing": (29.5630, 106.5516),
        "Fujian": (26.0789, 117.9874),
        "Gansu": (36.0611, 103.8343),
        "Guangdong": (23.3790, 113.7633),
        "Guangxi": (22.8240, 108.3650),
        "Guizhou": (26.5982, 106.7074),
        "Hainan": (20.0174, 110.3492),
        "Hebei": (38.0428, 114.5149),
        "Heilongjiang": (45.8038, 126.5349),
        "Henan": (34.7657, 113.7536),
        "Hubei": (30.5950, 114.2999),
        "Hunan": (28.1127, 112.9838),
        "Inner Mongolia": (40.8175, 111.6708),
        "Jiangsu": (32.0603, 118.7969),
        "Jiangxi": (28.6765, 115.8922),
        "Jilin": (43.8965, 125.3268),
        "Liaoning": (41.7968, 123.4290),
        "Ningxia": (38.4680, 106.2732),
        "Qinghai": (36.6232, 101.7782),
        "Shaanxi": (34.3416, 108.9398),
        "Shandong": (36.6685, 116.9974),
        "Shanghai": (31.2304, 121.4737),
        "Shanxi": (37.8735, 112.5624),
        "Sichuan": (30.5728, 104.0668),
        "Tianjin": (39.3434, 117.3616),
        "Tibet": (29.6469, 91.1175),
        "Xinjiang": (43.7930, 87.6277),
        "Yunnan": (25.0438, 102.7097),
        "Zhejiang": (30.2741, 120.1551),
    }

    # 3. 数据验证
    print("=" * 70)
    print("步骤3: 数据验证")
    print("=" * 70)
    coord_cities, emission_cities = set(city_coords.keys()), set(avg.keys())
    valid_cities = list(coord_cities.intersection(emission_cities))
    print(f"有效双维度城市数: {len(valid_cities)} 个")
    if not valid_cities:
        raise ValueError("无有效城市数据！")

    # 4. 特征构建
    print("\n" + "=" * 70)
    print("步骤4: 双维度特征构建")
    print("=" * 70)
    all_years = sorted(list(set(y for city in valid_cities for y in avg[city].keys())))
    cities, double_dim_features, latitudes, longitudes = [], [], [], []
    for city in valid_cities:
        cities.append(city)
        lat, lon = city_coords[city]
        latitudes.append(lat)
        longitudes.append(lon)
        emission_features = [avg[city].get(year, np.nan) for year in all_years]
        # 简单处理缺失值，用该城市排放均值填充
        emission_features = (
            pd.Series(emission_features)
            .fillna(pd.Series(emission_features).mean())
            .tolist()
        )
        double_dim_features.append([lat, lon] + emission_features)

    # 5. 数据标准化
    print("\n" + "=" * 70)
    print("步骤5: 双维度数据标准化")
    print("=" * 70)
    geo_features = [feat[:2] for feat in double_dim_features]
    emission_features = [feat[2:] for feat in double_dim_features]
    geo_scaled = StandardScaler().fit_transform(geo_features)
    emission_scaled = StandardScaler().fit_transform(emission_features)
    double_dim_features_scaled = np.hstack([geo_scaled, emission_scaled])
    print("地理特征与碳排放特征已分别标准化并合并。")

    # 6. KMeans聚类
    print("\n" + "=" * 70)
    print("步骤6: 双维度KMeans聚类")
    print("=" * 70)
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=30, max_iter=500)
    cluster_labels = kmeans.fit_predict(double_dim_features_scaled)

    # 6a. 聚类效果多维度评估
    print("\n" + "=" * 70)
    print("步骤6a: 聚类效果多维度评估")
    print("=" * 70)
    sil_score = silhouette_score(double_dim_features_scaled, cluster_labels)
    ch_score = calinski_harabasz_score(double_dim_features_scaled, cluster_labels)
    db_score = davies_bouldin_score(double_dim_features_scaled, cluster_labels)
    print(f"  - 轮廓系数 (Silhouette Score): {sil_score:.3f} (越高越好, >0.5算好)")
    print(
        f"  - 卡林斯基-哈拉巴斯指数 (Calinski-Harabasz Score): {ch_score:.1f} (越高越好)"
    )
    print(
        f"  - 戴维斯-布尔丁指数 (Davies-Bouldin Score): {db_score:.3f} (越低越好, <1.0算好)"
    )

    # 7. 聚类结果整理与统计
    print("\n" + "=" * 70)
    print("步骤7: 聚类结果整理与统计")
    print("=" * 70)
    cluster_result = pd.DataFrame(
        {
            "城市名称": cities,
            "纬度（经纬度字典）": latitudes,
            "经度（经纬度字典）": longitudes,
            "聚类标签": cluster_labels,
            **{
                f"{year}年排放均值": [ef[i] for ef in emission_features]
                for i, year in enumerate(all_years)
            },
        }
    )

    # 打印统计摘要
    for cluster_id in range(n_clusters):
        cluster_data = cluster_result[cluster_result["聚类标签"] == cluster_id]
        if cluster_data.empty:
            continue
        avg_lat, avg_lon = (
            cluster_data["纬度（经纬度字典）"].mean(),
            cluster_data["经度（经纬度字典）"].mean(),
        )
        avg_emission_2019 = (
            cluster_data["2019年排放均值"].mean()
            if "2019年排放均值" in cluster_data
            else "N/A"
        )
        print(f"\n聚类{cluster_id} (共 {len(cluster_data)} 个城市):")
        print(f"  - 地理中心: (Lat: {avg_lat:.2f}, Lon: {avg_lon:.2f})")
        print(
            f"  - 2019年平均排放: {avg_emission_2019:.2f}"
            if isinstance(avg_emission_2019, float)
            else f"  - 2019年平均排放: {avg_emission_2019}"
        )
        print(f"  - 包含城市: {', '.join(cluster_data['城市名称'].tolist())}")

    # 8. 可视化
    # 8a. 评估指标可视化 (来自代码B的优点)
    draw_evaluation_plots(sil_score, ch_score, db_score)
    # 8b. 地理分布可视化 (来自代码A的优点)
    draw_geographic_map(cluster_result)

    # 9. 结果导出
    output_path = "中国省份双维度聚类结果（经纬度+碳排放）.csv"
    cluster_result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 70)
    print(f"最终聚类结果已保存至: '{output_path}'")
    print("=" * 70)
