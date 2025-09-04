from kmeans0 import kmeans_clustering_world_data
from kmeans1 import kmeans_clustering
from kmeans2 import create_plots
from ai_mod import ai_analysis


def main():
    kmeans_clustering_world_data()
    input("\n第一个业务执行完毕，按Enter继续执行下一个业务...")

    kmeans_clustering()
    input("\n第二个业务执行完毕，按Enter继续执行下一个业务...")

    create_plots()
    input("\n第三个业务执行完毕，按Enter继续执行最后一个业务...")

    ai_analysis()
    input("\n所有业务执行完成，按Enter退出程序...")


if __name__ == "__main__":
    main()
