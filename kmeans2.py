import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from data import CHINA_DATA

warnings.filterwarnings("ignore")

# 设置matplotlib支持中文
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def simple_kmeans(X, k=3, max_iters=100):
    """简单的K-means实现"""
    np.random.seed(42)

    # 随机初始化聚类中心
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # 计算每个点到聚类中心的距离
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        # 分配到最近的聚类中心
        labels = np.argmin(distances, axis=0)

        # 更新聚类中心
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # 检查收敛
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


def prepare_city_features(data):
    """准备城市特征"""
    print("准备城市特征...")

    city_features = []

    for city in data["city_name"].unique():
        city_data = data[data["city_name"] == city]

        # 基本统计特征
        total_emission = city_data["value"].sum()
        mean_emission = city_data["value"].mean()
        std_emission = city_data["value"].std() if len(city_data) > 1 else 0

        # 地理特征
        lat = city_data["lat"].iloc[0] if "lat" in city_data.columns else 0
        lon = city_data["lon"].iloc[0] if "lon" in city_data.columns else 0

        # 时间特征
        years_count = city_data["year"].nunique()

        # 趋势计算
        yearly_sum = city_data.groupby("year")["value"].sum()
        if len(yearly_sum) > 1:
            years = yearly_sum.index.values
            values = yearly_sum.values
            # 简单线性回归计算趋势
            x_mean = np.mean(years)
            y_mean = np.mean(values)
            slope = np.sum((years - x_mean) * (values - y_mean)) / np.sum(
                (years - x_mean) ** 2
            )
        else:
            slope = 0

        city_features.append(
            [total_emission, mean_emission, std_emission, lat, lon, years_count, slope]
        )

    feature_names = [
        "total_emission",
        "mean_emission",
        "std_emission",
        "lat",
        "lon",
        "years_count",
        "trend_slope",
    ]

    cities = data["city_name"].unique()

    return np.array(city_features), cities, feature_names


def normalize_features(X):
    """标准化特征"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)


def analyze_and_visualize(data):
    """分析并可视化（特征说明增强版）"""
    print("开始分析...")

    # 准备特征
    X, cities, feature_names = prepare_city_features(data)
    print(f"特征矩阵形状: {X.shape}")

    # 特征详细说明 (只在此处详细打印一次)
    feature_descriptions = {
        "total_emission": "总排放量 (万吨CO₂)",
        "mean_emission": "平均年排放量 (万吨CO₂/年)",
        "std_emission": "排放量标准差 (波动性)",
        "lat": "纬度 (地理位置-南北)",
        "lon": "经度 (地理位置-东西)",
        "years_count": "数据年份数量 (数据完整性)",
        "trend_slope": "排放趋势斜率 (时间变化率)",
    }

    print("\n🎯 聚类使用的7个特征详细说明:")
    print("=" * 50)
    for i, (name, desc) in enumerate(feature_descriptions.items()):
        print(f"特征{i + 1}: {name:<15} - {desc}")
    print("=" * 50)
    print("📌 K-means聚类将基于以上7个特征的综合相似性进行")
    print("📌 稍后生成的图表将从不同维度展示聚类结果\n")

    # 标准化特征
    X_normalized = normalize_features(X)

    # 寻找最优K值
    print("寻找最优K值 (肘部法则)...")
    k_range = range(2, min(8, len(cities) // 2))
    inertias = []

    for k in k_range:
        labels, centroids = simple_kmeans(X_normalized, k=k)
        inertia = sum(
            [
                np.sum((X_normalized[labels == i] - centroids[i]) ** 2)
                for i in range(k)
                if np.sum(labels == i) > 0
            ]
        )
        inertias.append(inertia)
        print(f"  K={k}: 惯性={inertia:.2f}")

    # 可视化肘部法则
    plt.figure(figsize=(12, 6))
    plt.plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
    plt.xlabel("聚类数量 (K)", fontsize=12)
    plt.ylabel("聚类惯性值 (Within-cluster Sum of Squares)", fontsize=12)
    plt.title(
        "肘部法则 - 基于7维特征的最优K值选择\n"
        + "特征: 排放量统计(3) + 地理位置(2) + 时间特征(1) + 趋势特征(1)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    textstr = "聚类特征:\n• 总排放量 & 平均排放量 & 排放波动性\n• 纬度 & 经度\n• 数据年份数\n• 排放趋势斜率"
    props = dict(boxstyle="round", facecolor="lightblue", alpha=0.8)
    plt.text(
        0.02,
        0.98,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    for i, (k, inertia) in enumerate(zip(k_range, inertias)):
        plt.annotate(
            f"{inertia:.1f}",
            (k, inertia),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()
    plt.show()

    # 使用最优K值进行聚类
    optimal_k = 4
    print(f"\n根据肘部法则，选择K={optimal_k}进行最终聚类...")

    labels, centroids = simple_kmeans(X_normalized, k=optimal_k)

    # 创建结果DataFrame
    results_df = pd.DataFrame(X, columns=feature_names)
    results_df["city_name"] = cities
    results_df["cluster"] = labels

    # 计算聚类统计
    cluster_stats = {}
    for i in range(optimal_k):
        mask = labels == i
        cluster_data = X[mask]
        cluster_cities = cities[mask]

        cluster_stats[i] = {
            "count": len(cluster_cities),
            "cities": cluster_cities[:3].tolist(),
            "total_emission_mean": cluster_data[:, 0].mean(),
            "mean_emission_mean": cluster_data[:, 1].mean(),
            "std_emission_mean": cluster_data[:, 2].mean(),
            "lat_mean": cluster_data[:, 3].mean(),
            "lon_mean": cluster_data[:, 4].mean(),
            "years_count_mean": cluster_data[:, 5].mean(),
            "trend_mean": cluster_data[:, 6].mean(),
            "trend_desc": (
                "上升趋势"
                if cluster_data[:, 6].mean() > 0.1
                else "下降趋势" if cluster_data[:, 6].mean() < -0.1 else "相对稳定"
            ),
        }

    # 创建增强版可视化
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        f"基于7维特征的城市碳排放K-means聚类分析 (K={optimal_k})\n"
        + "聚类特征: 排放统计特征 + 地理位置特征 + 时间特征 + 趋势特征",
        fontsize=18,
        fontweight="bold",
        y=0.95,
    )

    colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    # 1. 地理位置特征展示 (特征4,5: 纬度、经度)
    ax1 = plt.subplot(2, 3, 1)
    for i in range(optimal_k):
        mask = labels == i
        ax1.scatter(
            X[mask, 4],
            X[mask, 3],
            c=[colors[i]],
            label=f'聚类{i} ({cluster_stats[i]["count"]}城市)',
            s=80,
            alpha=0.7,
            marker=markers[i],
            edgecolors="black",
            linewidth=0.5,
        )
    centroids_original = np.array(
        [X[labels == i].mean(axis=0) for i in range(optimal_k)]
    )
    ax1.scatter(
        centroids_original[:, 4],
        centroids_original[:, 3],
        c="red",
        marker="X",
        s=300,
        linewidths=2,
        label="聚类中心",
        edgecolors="black",
    )
    ax1.set_xlabel("🌍 特征5: 经度 (Longitude)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("🌍 特征4: 纬度 (Latitude)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "地理位置特征聚类展示", fontsize=12, fontweight="bold", color="darkblue"
    )
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 排放量统计特征展示 (特征1,2: 总排放量、平均排放量)
    ax2 = plt.subplot(2, 3, 2)
    for i in range(optimal_k):
        mask = labels == i
        ax2.scatter(
            X[mask, 0],
            X[mask, 1],
            c=[colors[i]],
            label=f'聚类{i}: {cluster_stats[i]["trend_desc"]}',
            s=80,
            alpha=0.7,
            marker=markers[i],
            edgecolors="black",
            linewidth=0.5,
        )
    ax2.scatter(
        centroids_original[:, 0],
        centroids_original[:, 1],
        c="red",
        marker="X",
        s=300,
        linewidths=2,
        label="聚类中心",
        edgecolors="black",
    )
    ax2.set_xlabel("📊 特征1: 总排放量 (万吨CO₂)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("📊 特征2: 平均排放量 (万吨CO₂/年)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "排放量统计特征聚类展示", fontsize=12, fontweight="bold", color="darkgreen"
    )
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. 聚类数量统计
    ax3 = plt.subplot(2, 3, 3)
    unique, counts = np.unique(labels, return_counts=True)
    bars = ax3.bar(
        unique,
        counts,
        color=colors[: len(unique)],
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.3,
            f"{count}城市\n({count / len(cities) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax3.set_xlabel("聚类编号", fontsize=12, fontweight="bold")
    ax3.set_ylabel("城市数量", fontsize=12, fontweight="bold")
    ax3.set_title("聚类规模分布", fontsize=12, fontweight="bold", color="darkorange")
    ax3.set_xticks(unique)
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. 趋势特征展示 (特征7: 趋势斜率)
    ax4 = plt.subplot(2, 3, 4)
    trend_means = [cluster_stats[i]["trend_mean"] for i in range(optimal_k)]
    bars = ax4.bar(
        range(optimal_k),
        trend_means,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    for i, (bar, trend) in enumerate(zip(bars, trend_means)):
        height = bar.get_height()
        y_pos = height + 0.002 if height >= 0 else height - 0.004
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f'{trend:.3f}\n{cluster_stats[i]["trend_desc"]}',
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=9,
            fontweight="bold",
        )
    ax4.axhline(y=0, color="red", linestyle="--", alpha=0.8, linewidth=2)
    ax4.set_xlabel("聚类编号", fontsize=12, fontweight="bold")
    ax4.set_ylabel("📈 特征7: 趋势斜率 (万吨CO₂/年²)", fontsize=12, fontweight="bold")
    ax4.set_title(
        "排放趋势特征聚类展示", fontsize=12, fontweight="bold", color="darkred"
    )
    ax4.grid(True, alpha=0.3)

    # 5. 排放波动性特征展示 (特征3: 标准差)
    ax5 = plt.subplot(2, 3, 5)
    std_means = [cluster_stats[i]["std_emission_mean"] for i in range(optimal_k)]
    bars = ax5.bar(
        range(optimal_k),
        std_means,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    for bar, std_val in zip(bars, std_means):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.02,
            f"{std_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax5.set_xlabel("聚类编号", fontsize=12, fontweight="bold")
    ax5.set_ylabel("📊 特征3: 排放标准差 (波动性)", fontsize=12, fontweight="bold")
    ax5.set_title(
        "排放波动性特征聚类展示", fontsize=12, fontweight="bold", color="darkviolet"
    )
    ax5.grid(True, alpha=0.3)

    # 6. 数据完整性特征展示 (特征6: 年份数量)
    ax6 = plt.subplot(2, 3, 6)
    years_means = [cluster_stats[i]["years_count_mean"] for i in range(optimal_k)]
    bars = ax6.bar(
        range(optimal_k),
        years_means,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    for bar, years_val in zip(bars, years_means):
        height = bar.get_height()
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.02,
            f"{years_val:.1f}年",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax6.set_xlabel("聚类编号", fontsize=12, fontweight="bold")
    ax6.set_ylabel("📅 特征6: 数据年份数量", fontsize=12, fontweight="bold")
    ax6.set_title(
        "数据完整性特征聚类展示", fontsize=12, fontweight="bold", color="darkcyan"
    )
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 创建7维特征综合展示图
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(polar=True))

    feature_labels_chinese = list(
        v.split(" ")[0] for v in feature_descriptions.values()
    )
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False).tolist()
    angles += angles[:1]

    for i in range(optimal_k):
        cluster_data = X_normalized[labels == i]
        if len(cluster_data) > 0:
            radar_values = cluster_data.mean(axis=0).tolist()
            radar_values += radar_values[:1]
            ax.plot(
                angles,
                radar_values,
                "o-",
                linewidth=3,
                label=f'聚类{i} ({cluster_stats[i]["count"]}城市)',
                color=colors[i],
                alpha=0.8,
                markersize=8,
            )
            ax.fill(angles, radar_values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels_chinese, fontsize=12, fontweight="bold")
    ax.set_title(
        "7维特征综合雷达图\n展示各聚类在所有特征上的表现模式 (0为平均水平)",
        fontsize=16,
        fontweight="bold",
        pad=30,
    )
    ax.legend(bbox_to_anchor=(1.2, 1), loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # (***优化点***) 输出精简后的详细特征聚类说明
    print("\n" + "🎯" * 30)
    print("特征聚类详细分析报告")
    print("🎯" * 30)
    print(f"\n📋 聚类方法: K-means算法 (K={optimal_k})")
    print("📋 聚类依据: 基于已说明的7个特征进行综合分析。")
    print("\n📊 聚类结果解读:")
    print("-" * 60)

    for i in range(optimal_k):
        print(f"\n🏙️  聚类 {i} 特征画像 (详细数据对比请参考雷达图)")
        print("-" * 50)
        print(f"📍 城市数量: {cluster_stats[i]['count']} 个")
        print(f"📍 代表城市: {', '.join(cluster_stats[i]['cities'])} (及其他)")

        # (***优化点***) 移除重复的7维特征罗列，只保留核心总结
        print(f"\n🎯 核心特征总结:")
        emission_level = (
            "高排放"
            if cluster_stats[i]["mean_emission_mean"] > 30
            else "中排放" if cluster_stats[i]["mean_emission_mean"] > 10 else "低排放"
        )
        stability = (
            "波动大" if cluster_stats[i]["std_emission_mean"] > 20 else "相对稳定"
        )
        print(
            f"  • 排放水平: {emission_level} (均值: {cluster_stats[i]['mean_emission_mean']:.2f})"
        )
        print(
            f"  • 排放稳定性: {stability} (标准差: {cluster_stats[i]['std_emission_mean']:.2f})"
        )
        print(
            f"  • 发展趋势: {cluster_stats[i]['trend_desc']} (斜率: {cluster_stats[i]['trend_mean']:.3f})"
        )
        print(
            f"  • 地理分布中心: 纬度{cluster_stats[i]['lat_mean']:.1f}°, 经度{cluster_stats[i]['lon_mean']:.1f}°"
        )

    return results_df


# ... (从 linear_trend_with_confidence 到 ensemble_prediction 的函数保持不变) ...
def linear_trend_with_confidence(time_series, future_years, confidence=0.95):
    """线性趋势预测（带置信区间）"""
    x = np.arange(len(time_series))
    y = time_series["value"].values
    if len(x) < 2:
        return {"method": "Linear Trend (Not enough data)"}

    x_mean, y_mean = np.mean(x), np.mean(y)
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean

    future_x = np.arange(len(time_series), len(time_series) + future_years)
    future_pred = slope * future_x + intercept

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    residuals = y - y_pred
    mse = np.mean(residuals**2)
    se = np.sqrt(
        mse * (1 + 1 / len(x) + (future_x - x_mean) ** 2 / np.sum((x - x_mean) ** 2))
    )
    margin_error = 1.96 * se

    return {
        "predictions": future_pred,
        "upper_bound": future_pred + margin_error,
        "lower_bound": future_pred - margin_error,
        "r_squared": r_squared,
        "slope": slope,
        "method": "Linear Trend",
    }


def polynomial_trend_prediction(time_series, future_years, degree=2):
    """多项式趋势预测"""
    x = np.arange(len(time_series))
    y = time_series["value"].values
    if len(x) < 4:
        return {"method": "Polynomial (Not enough data)"}

    best_degree, best_score = 1, -np.inf
    for d in range(1, min(4, len(time_series) // 3)):
        coeffs = np.polyfit(x, y, d)
        poly_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - poly_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        adj_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - d - 1)
        if adj_r_squared > best_score:
            best_score, best_degree = adj_r_squared, d

    coeffs = np.polyfit(x, y, best_degree)
    future_x = np.arange(len(time_series), len(time_series) + future_years)
    future_pred = np.polyval(coeffs, future_x)

    return {
        "predictions": future_pred,
        "degree": best_degree,
        "r_squared": best_score,
        "method": f"Polynomial (d={best_degree})",
    }


def moving_average_prediction(time_series, future_years, window=None):
    """移动平均预测"""
    values = time_series["value"].values
    if len(values) < 2:
        return {"method": "Moving Average (Not enough data)"}

    if window is None:
        window = min(max(3, len(values) // 4), 12)
    if len(values) < window:
        window = len(values)

    moving_avg = np.convolve(values, np.ones(window) / window, mode="valid")

    if len(moving_avg) > 1:
        x = np.arange(len(moving_avg))
        x_mean, y_mean = np.mean(x), np.mean(moving_avg)
        slope = np.sum((x - x_mean) * (moving_avg - y_mean)) / np.sum((x - x_mean) ** 2)

        last_avg = moving_avg[-1]
        future_pred = [last_avg + slope * (i + 1) for i in range(future_years)]
        return {
            "predictions": np.array(future_pred),
            "window_size": window,
            "trend_slope": slope,
            "method": f"MA (win={window})",
        }
    else:
        return {
            "predictions": np.full(future_years, values[-1]),
            "method": "Simple Average",
        }


def exponential_smoothing_prediction(time_series, future_years, alpha=0.3):
    """指数平滑预测"""
    values = time_series["value"].values
    if len(values) < 2:
        return {"method": "Exp. Smoothing (Not enough data)"}

    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[i - 1])
    smoothed = np.array(smoothed)

    trend = smoothed[-1] - smoothed[-2] if len(smoothed) > 1 else 0
    last_value = smoothed[-1]
    future_pred = [last_value + trend * (i + 1) for i in range(future_years)]

    return {
        "predictions": np.array(future_pred),
        "alpha": alpha,
        "trend": trend,
        "method": f"Exp. Smooth (α={alpha:.2f})",
    }


def ensemble_prediction(predictions, future_years):
    """集成预测（多模型平均）"""
    valid_predictions, weights, methods = [], [], []
    for method, pred_data in predictions.items():
        if method != "ensemble" and "predictions" in pred_data:
            valid_predictions.append(pred_data["predictions"])
            methods.append(pred_data["method"])
            weights.append(max(pred_data.get("r_squared", 0.5), 0.1))

    if not valid_predictions:
        return {"method": "Ensemble (No valid preds)"}

    weights = np.array(weights) / np.sum(weights)
    ensemble_pred = np.zeros(future_years)
    for pred, weight in zip(valid_predictions, weights):
        if len(pred) == future_years:
            ensemble_pred += weight * pred

    return {
        "predictions": ensemble_pred,
        "weights": dict(zip(methods, weights)),
        "method": "Ensemble Average",
    }


def perform_multiple_predictions(time_series, future_years):
    """执行多种预测方法"""
    predictions = {
        "linear": linear_trend_with_confidence(time_series, future_years),
        "polynomial": polynomial_trend_prediction(time_series, future_years),
        "moving_average": moving_average_prediction(time_series, future_years),
        "exponential": exponential_smoothing_prediction(time_series, future_years),
    }
    predictions["ensemble"] = ensemble_prediction(predictions, future_years)
    return predictions


def enhanced_prediction(data, results_df, future_years=5):
    """增强版趋势预测"""
    print(f"\n未来{future_years}年增强趋势预测:")
    print("=" * 80)

    for cluster_id in sorted(results_df["cluster"].unique()):
        cluster_cities = results_df[results_df["cluster"] == cluster_id][
            "city_name"
        ].tolist()
        cluster_data = data[data["city_name"].isin(cluster_cities)]

        print(f"\n聚类 {cluster_id} - 包含 {len(cluster_cities)} 个城市")
        print("-" * 60)

        time_series = cluster_data.groupby("year").agg({"value": "sum"}).reset_index()
        time_series["date"] = pd.to_datetime(time_series["year"], format="%Y")
        time_series = time_series.sort_values("date").reset_index(drop=True)

        if len(time_series) < 3:
            print(f"数据点太少 ({len(time_series)} 个)，跳过预测。")
            continue

        predictions = perform_multiple_predictions(time_series, future_years)
        visualize_enhanced_predictions(
            time_series, predictions, cluster_id, future_years
        )
        print_prediction_summary(predictions, time_series)


def visualize_enhanced_predictions(time_series, predictions, cluster_id, future_years):
    """可视化增强预测结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"聚类 {cluster_id} - 多模型碳排放预测分析", fontsize=16)

    historical_dates = time_series["date"]
    historical_values = time_series["value"]
    last_date = historical_dates.iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(years=1), periods=future_years, freq="YS"
    )

    # 1. 线性趋势
    ax1 = axes[0, 0]
    ax1.plot(historical_dates, historical_values, "b-", label="历史数据", linewidth=2)
    if "linear" in predictions and "predictions" in predictions["linear"]:
        linear_pred = predictions["linear"]
        ax1.plot(
            future_dates,
            linear_pred["predictions"],
            "r--",
            label=f"线性预测 (R²={linear_pred['r_squared']:.3f})",
            linewidth=2,
        )
        ax1.fill_between(
            future_dates,
            linear_pred["lower_bound"],
            linear_pred["upper_bound"],
            alpha=0.3,
            color="red",
            label="95%置信区间",
        )
    ax1.set_title("线性趋势预测")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # 2. 多项式与移动平均
    ax2 = axes[0, 1]
    ax2.plot(historical_dates, historical_values, "b-", label="历史数据", linewidth=2)
    if "polynomial" in predictions and "predictions" in predictions["polynomial"]:
        poly_pred = predictions["polynomial"]
        ax2.plot(
            future_dates,
            poly_pred["predictions"],
            "g--",
            label=f"{poly_pred['method']}",
            linewidth=2,
        )
    if (
        "moving_average" in predictions
        and "predictions" in predictions["moving_average"]
    ):
        ma_pred = predictions["moving_average"]
        ax2.plot(
            future_dates,
            ma_pred["predictions"],
            "m--",
            label=f"{ma_pred['method']}",
            linewidth=2,
        )
    ax2.set_title("多项式与移动平均预测")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    # 3. 指数平滑与集成
    ax3 = axes[1, 0]
    ax3.plot(historical_dates, historical_values, "b-", label="历史数据", linewidth=2)
    if "exponential" in predictions and "predictions" in predictions["exponential"]:
        exp_pred = predictions["exponential"]
        ax3.plot(
            future_dates,
            exp_pred["predictions"],
            "orange",
            linestyle="--",
            label=f"{exp_pred['method']}",
            linewidth=2,
        )
    ax3.set_title("指数平滑预测")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis="x", rotation=45)

    # 4. 集成预测对比
    ax4 = axes[1, 1]
    ax4.plot(historical_dates, historical_values, "b-", label="历史数据", linewidth=3)
    colors = {"Linear": "red", "Polynomial": "green", "MA": "purple", "Exp.": "orange"}
    for method, pred_data in predictions.items():
        if "predictions" in pred_data and method != "ensemble":
            key_part = pred_data["method"].split(" ")[0]
            ax4.plot(
                future_dates,
                pred_data["predictions"],
                color=colors.get(key_part, "gray"),
                linestyle="--",
                alpha=0.7,
                label=pred_data["method"],
                linewidth=1.5,
            )
    if "ensemble" in predictions and "predictions" in predictions["ensemble"]:
        ensemble_pred = predictions["ensemble"]
        ax4.plot(
            future_dates,
            ensemble_pred["predictions"],
            "black",
            linestyle="-",
            label="集成预测",
            linewidth=3,
        )
    ax4.set_title("所有预测方法对比")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def print_prediction_summary(predictions, time_series):
    """(***优化点***) 打印精简后的预测摘要, 聚焦于集成模型"""
    historical_mean = time_series["value"].mean()
    print("预测摘要 (基于多模型集成):")
    print("-" * 50)

    ensemble_pred_data = predictions.get("ensemble")
    if ensemble_pred_data and "predictions" in ensemble_pred_data:
        future_mean = np.mean(ensemble_pred_data["predictions"])
        change_percent = (
            ((future_mean - historical_mean) / historical_mean) * 100
            if historical_mean != 0
            else 0
        )

        print(
            f"  集成模型未来 {len(ensemble_pred_data['predictions'])} 年平均预测值: {future_mean:.2f}"
        )
        print(f"  相比历史均值的变化: {change_percent:+.1f}%")

        if "weights" in ensemble_pred_data:
            weights_str = ", ".join(
                [
                    f"{k.split(' ')[0]}: {v:.1%}"
                    for k, v in ensemble_pred_data["weights"].items()
                ]
            )
            print(f"  各模型权重: {weights_str}")
    else:
        print("  未能生成有效的集成预测结果。")
    print()


# ==================== 主执行函数 ====================


def create_plots():
    """主函数"""
    print("函数开始执行，稍安勿躁...")
    print("开始碳排放K-means聚类分析...")

    # 检查数据
    print(f"数据形状: {CHINA_DATA.shape}")
    print(f"城市数量: {CHINA_DATA['city_name'].nunique()}")

    # 执行聚类分析
    results = analyze_and_visualize(CHINA_DATA)

    # 执行增强版趋势预测
    enhanced_prediction(CHINA_DATA, results, future_years=5)

    print("\n分析完成！")


# 如果您想直接运行此文件进行测试，可以取消以下代码的注释
# if __name__ == '__main__':
#     create_plots()
