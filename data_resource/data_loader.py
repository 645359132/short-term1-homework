import pandas as pd
DATA_URL = 'https://datas.carbonmonitor.org/API/downloadFullDataset.php?source=carbon_cities'


class DataLoader(object):
    """
    用于加载、存储数据集的工具类。
    """
    def __init__(self):
        self.data = None
        self._load_data_from_url()

    def _load_data_from_url(self):
        """
        直接从URL加载CSV数据到Pandas DataFrame。

        参数:
        url (str): CSV文件的URL。

        返回:
        pandas.DataFrame: 加载后的数据框，如果失败则返回None。
        """
        print(f"正在从 {DATA_URL} 加载数据...")
        try:
            self.data = pd.read_csv(DATA_URL)
            print("数据加载成功！")
            print("数据维度:", self.data.shape)
            print("前5行数据:")
            print(self.data.head())
            return self.data
        except Exception as e:
            print(f"从URL加载数据失败: {e}")
            return None

    def trans_data_to_dict(self):
        """
        将DataFrame数据转换为字典。

        返回:
        dict: 转换后的字典，如果失败则返回None。
        """
        try:
            data_dict = self.data.to_dict(orient='records')
            return data_dict
        except Exception as e:
            print(f"转换数据为字典失败: {e}")
            return None

    def trans_data_to_list(self):
        """
        将DataFrame数据转换为列表。

        返回:
        list: 转换后的列表，如果失败则返回None。
        """
        try:
            data_list = self.data.values.tolist()
            return data_list
        except Exception as e:
            print(f"转换数据为列表失败: {e}")
            return None

