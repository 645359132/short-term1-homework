import pandas as pd

WORLD_MAIN_CITIES_DATA_URL = (
    "https://datas.carbonmonitor.org/API/downloadFullDataset.php?source=carbon_cities"
)
CHINA_PROVINCE_DATA_URL = (
    "https://datas.carbonmonitor.org/API/downloadFullDataset.php?source=carbon_china"
)


class DataLoader(object):
    """
    用于加载、存储数据集的工具类。
    """

    def __init__(self, url):
        self.data = None
        self._load_data_from_url(url)
        self.data = self._data_clean()
        self.data = self._add_time_features()  # 如果不需要时间特征，可以注释掉这一行

    def _load_data_from_url(self, url):
        """
        直接从URL加载CSV数据到Pandas DataFrame

        返回:
        pandas.DataFrame: 加载后的数据框，如果失败则返回None。
        """
        print(f"正在从 {url} 加载数据...")
        try:
            self.data = pd.read_csv(url)
            print("数据加载成功！")
            print("数据维度:", self.data.shape)
            print("前5行数据:")
            print(self.data.head())
            return self.data
        except Exception as e:
            print(f"从URL加载数据失败: {e}")
            return None

    def _data_clean(self):
        """
        数据清洗，包括缺失值处理、异常值处理等。
        :return:
        """
        df = self.data.copy()
        # --- 步骤 0: 初步探查与信息打印 ---
        print("原始数据维度:", df.shape)
        print("原始列名:", df.columns.tolist())

        # --- 步骤 1: 删除完全无用的列 ---
        unnamed_cols = [col for col in df.columns if "unnamed:" in str(col)]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            print(f"\n删除了无用的列: {unnamed_cols}")

        # --- 步骤 2: 标准化和重命名列名 ---
        # 将现有列名转换为小写
        df.columns = [str(col).lower() for col in df.columns]

        # 明确地重命名关键列，以统一后续代码
        rename_dict = {"city": "city_name", "state": "city_name"}
        df = df.rename(columns=rename_dict)
        print("\n标准化/重命名后的列名:", df.columns.tolist())

        # --- 步骤 3: 转换数据类型 ---
        print("\n正在转换 'date' 列为 datetime 类型...")
        try:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True)
            print("'date' 列成功转换为 datetime 类型。")
        except Exception as e:
            print(f"转换 'date' 列时发生错误: {e}. 请检查日期格式。")
            return None

        # --- 步骤 4: 处理缺失值 ---
        # 只使用数据中实际存在的列
        critical_cols = ["city_name", "date", "sector"]

        # 检查这些列是否真的存在，以防万一
        existing_critical_cols = [col for col in critical_cols if col in df.columns]
        if len(existing_critical_cols) != len(critical_cols):
            print(
                f"警告：某些关键列不存在！期望：{critical_cols}, 实际存在：{existing_critical_cols}"
            )

        original_rows = len(df)
        df = df.dropna(how="all")  # 删除所有列都为NaN的行
        rows_dropped = original_rows - len(df)
        if rows_dropped > 0:
            print(f"发现了并删除了 {rows_dropped} 条完全空的行。")
        else:
            print("未发现完全空的行。")

        # --- 步骤 4.2: 处理关键列的缺失值 ---
        print("\n检查并处理关键列的缺失值:")
        print(df.isnull().sum())

        # 填充value的缺失值
        df["value"] = df["value"].fillna(0)
        print("'value' 列的缺失值已用 0 填充。")

        # 删除关键信息缺失的行
        critical_cols = ["city_name", "date", "sector"]
        # value为0的点是时间序列的重要组成部分，因此不能删除。
        existing_critical_cols = [col for col in critical_cols if col in df.columns]

        original_rows = len(df)
        df = df.dropna(subset=existing_critical_cols)
        print(
            f"因 {existing_critical_cols} 列存在缺失值而删除了 {original_rows - len(df)} 行。"
        )

        # --- 步骤 5: 处理异常值 ---
        print("\n处理异常或不合理数据...")
        df = df[df["value"] >= 0]

        # --- 步骤 6: 清理文本数据 ---
        text_cols = ["city_name", "sector"]  # 同样，移除了 'country'
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].str.strip()

        # --- 步骤 7: 处理重复行 ---
        df = df.drop_duplicates()

        # 根据业务逻辑合并重复记录
        group_cols = ["date", "city_name", "sector", "lat", "lon"]
        # 确保所有分组列都存在
        existing_group_cols = [col for col in group_cols if col in df.columns]
        df = df.groupby(existing_group_cols).agg(value=("value", "sum")).reset_index()

        # --- 清洗流程结束 ---
        print("\n--- 数据清洗流程结束 ---")
        print("清洗后的数据维度:", df.shape)

        df = df.reset_index(drop=True)
        return df

    def _add_time_features(self):
        """
        添加时间特征列。
        """
        # 创建一个副本以避免修改原始传入的DataFrame
        df_featured = self.data.copy()

        # 使用 .dt 访问器来提取各种时间属性
        # .dt accessor provides access to datetime properties

        # 1. 基础时间单位 (Basic Time Units)
        df_featured["year"] = df_featured["date"].dt.year
        df_featured["month"] = df_featured["date"].dt.month
        df_featured["day"] = df_featured["date"].dt.day

        # 2. 星期相关特征 (Week-related Features)
        # dayofweek: 周一=0, 周日=6
        df_featured["day_of_week"] = df_featured["date"].dt.dayofweek
        # day_name: 'Monday', 'Tuesday', etc.
        df_featured["day_name"] = df_featured["date"].dt.day_name()
        # is_weekend: 方便进行布尔索引或作为模型特征
        df_featured["is_weekend"] = (df_featured["day_of_week"] >= 5).astype(
            int
        )  # 0 for weekday, 1 for weekend

        # 3. 年度周期性特征 (Annual Cyclical Features)
        # dayofyear: 一年中的第几天 (1-366)
        df_featured["day_of_year"] = df_featured["date"].dt.dayofyear
        # weekofyear: 一年中的第几周 (1-53)
        # isocalendar()返回一个包含year, week, day的DataFrame，我们只取week
        df_featured["week_of_year"] = (
            df_featured["date"].dt.isocalendar().week.astype(int)
        )
        # quarter: 季度 (1-4)
        df_featured["quarter"] = df_featured["date"].dt.quarter

        # 4. 月度周期性特征 (Monthly Cyclical Features)
        # is_month_start/end: 是否为月初/月末，可能与商业活动有关
        df_featured["is_month_start"] = df_featured["date"].dt.is_month_start.astype(
            int
        )
        df_featured["is_month_end"] = df_featured["date"].dt.is_month_end.astype(int)

        print(
            "已添加的特征列: 'year', 'month', 'day', 'day_of_week', 'day_name', 'is_weekend', 'day_of_year', 'week_of_year', 'quarter', 'is_month_start', 'is_month_end'"
        )
        print("--- 时间特征添加完成 ---")

        return df_featured

    def get_data(self):
        """
        获取DataFrame数据。

        返回:
        pandas.DataFrame: 数据框，如果失败则返回None。
        """
        return self.data

    def get_data_dict(self):
        """
        将DataFrame数据转换为字典并返回。

        返回:
        dict: 转换后的字典，如果失败则返回None。
        """
        try:
            data_dict = self.data.to_dict(orient="records")
            return data_dict
        except Exception as e:
            print(f"转换数据为字典失败: {e}")
            return None

    def get_data_list(self):
        """
        将DataFrame数据转换为列表并返回。

        返回:
        list: 转换后的列表，如果失败则返回None。
        """
        try:
            data_list = self.data.values.tolist()
            return data_list
        except Exception as e:
            print(f"转换数据为列表失败: {e}")
            return None


#data_obj = DataLoader(WORLD_MAIN_CITIES_DATA_URL)
#WORLD_DATA = data_obj.get_data()
# DATA就是清洗后的DataFrame数据
# print(DATA.head())
china_data_obj = DataLoader(CHINA_PROVINCE_DATA_URL)
CHINA_DATA = china_data_obj.get_data()
# print (CHINA_DATA.head())
