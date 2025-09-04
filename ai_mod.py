import matplotlib.pyplot as plt
from data import CHINA_DATA
import seaborn as sns
import pandas as pd
import requests
from datetime import datetime, timedelta
from openai import OpenAI
import json
import warnings

warnings.filterwarnings('ignore')

# 解决中文字体显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 设置图表样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 6)

# OpenAI客户端配置
client = OpenAI(
    base_url="https://api.juheai.top/v1",
    api_key="sk-anICTRGe4RzgI28acbCj5VZWdBnnqirO0WwFUKGvzFhrKpb6",
)


class CarbonEmissionAnalyzer:
    def __init__(self, data, openai_client):
        self.data = data
        self.client = openai_client
        self.analysis_results = {}

        # 尝试设置中文字体
        self.setup_chinese_font()

    def setup_chinese_font(self):
        """设置中文字体支持"""
        import matplotlib.font_manager as fm

        # 尝试不同的中文字体
        chinese_fonts = [
            'SimHei',  # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'SimSun',  # 宋体
            'KaiTi',  # 楷体
            'FangSong',  # 仿宋
            'STSong',  # 华文宋体
            'STKaiti',  # 华文楷体
            'STHeiti',  # 华文黑体
        ]

        available_fonts = [f.name for f in fm.fontManager.ttflist]

        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                print(f"✅ 成功设置中文字体: {font}")
                return

        # 如果没有找到中文字体，尝试使用系统默认
        print("⚠️  未找到常用中文字体，将使用系统默认字体")
        print("💡 如果中文显示异常，请安装中文字体或使用英文标签")

        # 提供备选方案：使用英文标签
        self.use_english_labels = True

    def filter_and_plot_data(self, city, sector):
        """筛选数据并绘制图表"""
        print(f"正在分析 {city} 的 {sector} 碳排放数据...")

        # 筛选数据
        filtered_data = self.data[
            (self.data['city_name'] == city) &
            (self.data['sector'] == sector)
            ].copy()

        if filtered_data.empty:
            print(f"未找到 {city} - {sector} 的数据")
            return None

        # 确保日期格式正确
        filtered_data['date'] = pd.to_datetime(filtered_data['date'])
        filtered_data = filtered_data.sort_values('date')

        # 绘制图表
        fig, ax = plt.subplots(figsize=(15, 6))

        # 绘制线图
        ax.plot(filtered_data['date'], filtered_data['value'],
                'b-', marker='o', markersize=3, alpha=0.7, linewidth=2)

        # 设置标题和标签（处理中文显示问题）
        try:
            ax.set_title(f'{city} - {sector} 碳排放趋势分析', fontsize=16, pad=20)
            ax.set_xlabel('日期', fontsize=12)
            ax.set_ylabel('CO2排放量 (吨)', fontsize=12)
        except:
            # 如果中文显示有问题，使用英文
            ax.set_title(f'{city} - {sector} Carbon Emission Trend Analysis', fontsize=16, pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('CO2 Emissions (tons)', fontsize=12)

        # 美化图表
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # 添加数据统计信息
        mean_value = filtered_data['value'].mean()
        ax.axhline(y=mean_value, color='r', linestyle='--', alpha=0.5, label=f'平均值: {mean_value:.2f}')
        ax.legend()

        plt.tight_layout()

        try:
            plt.show()
        except Exception as e:
            print(f"图表显示可能有问题: {e}")
            # 保存图片作为备选方案
            plt.savefig('carbon_emission_trend.png', dpi=300, bbox_inches='tight')
            print("📊 图表已保存为 carbon_emission_trend.png")

        return filtered_data

    def detect_anomalies(self, data, start_date='2022-01-25', end_date='2022-02-28'):
        """检测异常数据"""
        anomalous_period = data[
            (data['date'] >= start_date) &
            (data['date'] <= end_date)
            ]

        if anomalous_period.empty:
            return None

        # 找到最小值作为异常点
        anomaly = anomalous_period.loc[anomalous_period['value'].idxmin()]

        # 计算统计信息
        mean_value = data['value'].mean()
        std_value = data['value'].std()

        anomaly_info = {
            'date': anomaly['date'].strftime('%Y-%m-%d'),
            'value': float(anomaly['value']),
            'city': anomaly.get('city_name', 'Unknown'),
            'sector': anomaly.get('sector', 'Unknown'),
            'mean_value': mean_value,
            'std_value': std_value,
            'deviation': (float(anomaly['value']) - mean_value) / std_value
        }

        print("\n" + "🔍 异常数据检测结果".center(50, "="))
        print(f"  城市: {anomaly_info['city']}")
        print(f"  部门: {anomaly_info['sector']}")
        print(f"  异常日期: {anomaly_info['date']}")
        print(f"  异常值: {anomaly_info['value']:.2f} 吨 CO2")
        print(f"  数据均值: {anomaly_info['mean_value']:.2f} 吨 CO2")
        print(f"  标准差: {anomaly_info['std_value']:.2f}")
        print(f"  偏差程度: {anomaly_info['deviation']:.2f} 个标准差")
        print(f"  异常类型: {'异常低点' if anomaly_info['deviation'] < -1 else '正常范围'}")
        print("=" * 50)

        return anomaly_info

    def get_news_context(self, city, date, api_key=None):
        """获取新闻背景信息"""
        if not api_key:
            print("📰 未提供NewsAPI密钥，跳过新闻背景分析")
            return None

        def fetch_news(api_key, city, date):
            # 构建更好的查询词
            if city.lower() == 'beijing':
                query = '(Beijing OR 北京 OR "Chinese New Year" OR "Winter Olympics" OR COVID OR lockdown)'
            else:
                query = f'"{city}"'

            date_obj = datetime.strptime(date, '%Y-%m-%d')
            from_date = (date_obj - timedelta(days=5)).strftime('%Y-%m-%d')
            to_date = (date_obj + timedelta(days=5)).strftime('%Y-%m-%d')

            url = (
                'https://newsapi.org/v2/everything?'
                f'q={query}&'
                f'from={from_date}&to={to_date}&'
                f'language=en&'
                f'sortBy=relevancy&'
                f'pageSize=10&'
                f'apiKey={api_key}'
            )

            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()

                if data.get('status') == 'ok':
                    articles = data.get('articles', [])
                    headlines = [article['title'] for article in articles if article.get('title')]
                    return headlines[:8]  # 返回前8条
                else:
                    print(f"NewsAPI返回错误: {data.get('message', '未知错误')}")
                    return None
            except Exception as e:
                print(f"获取新闻失败: {e}")
                return None

        print(f"\n📰 正在搜索 {date} 附近的相关新闻...")
        headlines = fetch_news(api_key, city, date)

        if headlines:
            print(f"📰 找到 {len(headlines)} 条相关新闻:")
            for i, headline in enumerate(headlines, 1):
                print(f"  {i}. {headline}")
            return headlines
        else:
            print(f"📰 未找到 {date} 附近的相关新闻")
            return None

    def ai_analysis(self, anomaly_info, news_headlines=None, custom_question=None):
        """使用AI进行深度分析"""
        print("\n🤖 正在调用AI专家进行深度分析...")

        # 构建详细的分析上下文
        context = f"""
        【碳排放异常数据详细分析】

        基础信息:
        - 城市: {anomaly_info['city']}
        - 部门: {anomaly_info['sector']}
        - 异常日期: {anomaly_info['date']}
        - 异常值: {anomaly_info['value']:.2f} 吨 CO2
        - 数据均值: {anomaly_info['mean_value']:.2f} 吨 CO2
        - 偏差程度: {anomaly_info['deviation']:.2f} 个标准差

        时间背景分析:
        - 异常发生在2022年1-2月期间
        - 这个时期包含了中国春节假期（2022年1月31日-2月6日）
        - 北京2022年冬奥会（2月4日-20日）
        - COVID-19疫情防控措施可能仍在实施

        数据特征:
        - 这是一个显著的异常低点（低于均值{abs(anomaly_info['deviation']):.1f}个标准差）
        - 在交通运输部门出现如此低的排放值需要特别关注
        """

        if news_headlines:
            context += f"\n相关新闻背景:\n"
            for i, headline in enumerate(news_headlines, 1):
                context += f"{i}. {headline}\n"

        # 专业的系统提示词
        system_prompt = """你是一位资深的环境数据分析专家和碳排放研究学者，拥有丰富的城市交通碳排放分析经验。

请从以下专业角度深入分析这个碳排放异常：

🏛️ **政策制度因素**
- 疫情防控政策对交通流量的影响
- 环保限行政策和交通管制措施
- 春节期间的特殊交通政策

🎯 **重大事件影响**
- 北京冬奥会期间的交通管制和排放控制
- 春节假期对日常通勤模式的改变
- 突发公共卫生事件的影响

📊 **数据统计意义**
- 异常偏差程度的统计学解释
- 与历史同期数据的对比分析
- 数据质量和可靠性评估

🔬 **技术环境因素**
- 天气条件对交通出行的影响
- 新能源交通工具推广的可能影响
- 交通基础设施变化

💡 **专业建议**
- 数据验证和交叉检验建议
- 后续监测重点和方向
- 政策制定参考建议

请提供专业、深入、逻辑严密的分析报告。"""

        # 用户问题
        if custom_question:
            user_question = f"{context}\n\n【特殊分析需求】\n{custom_question}"
        else:
            user_question = f"{context}\n\n请对这个碳排放异常进行全面深入的专业分析，并提供有价值的洞察和建议。"

        try:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                temperature=0.6,  # 稍微降低随机性，提高专业性
                max_tokens=2000,  # 增加输出长度
                top_p=0.9
            )

            analysis_result = completion.choices[0].message.content
            self.analysis_results['ai_analysis'] = analysis_result
            self.analysis_results['anomaly_info'] = anomaly_info
            self.analysis_results['news_context'] = news_headlines

            print("\n" + "🎯 AI专家深度分析报告".center(80, "="))
            print(analysis_result)
            print("=" * 80)

            return analysis_result

        except Exception as e:
            error_msg = f"AI分析过程中出现错误: {str(e)}"
            print(f"\n❌ {error_msg}")
            print("💡 可能的原因：API密钥无效、网络连接问题、或服务暂时不可用")
            return error_msg

    def interactive_analysis(self):
        """交互式分析功能"""
        print("\n" + "🔬 交互式AI专家咨询系统".center(60, "="))
        print("💬 你可以询问关于碳排放分析的任何问题")
        print("💡 例如：")
        print("   - 这个异常可能的具体原因是什么？")
        print("   - 如何预防类似的异常情况？")
        print("   - 其他城市是否也有类似现象？")
        print("   - 这个数据对政策制定有什么启示？")
        print("\n📝 输入 'quit'、'exit' 或 'q' 退出咨询")
        print("=" * 60)

        conversation_history = []

        while True:
            try:
                user_question = input("\n🗣️  请输入你的问题: ").strip()

                if user_question.lower() in ['quit', 'exit', '退出', 'q', '']:
                    print("👋 感谢使用AI专家咨询系统，再见！")
                    break

                if len(user_question) < 3:
                    print("❓ 请输入更具体的问题（至少3个字符）")
                    continue

                print("🤖 AI专家正在思考...")

                # 构建对话上下文
                context_messages = [
                    {"role": "system", "content": """你是专业的碳排放数据分析专家。基于之前的分析结果，请回答用户的问题。
                    保持专业、准确、有见地。如果问题超出分析范围，请诚实说明并提供相关建议。"""}
                ]

                # 添加分析背景
                if self.analysis_results:
                    background = f"分析背景：{json.dumps(self.analysis_results, ensure_ascii=False, indent=2)}"
                    context_messages.append({"role": "assistant", "content": background})

                # 添加对话历史（最近3轮）
                for msg in conversation_history[-6:]:  # 保留最近3轮对话
                    context_messages.append(msg)

                # 添加当前问题
                context_messages.append({"role": "user", "content": user_question})

                completion = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=context_messages,
                    temperature=0.7,
                    max_tokens=1200,
                    top_p=0.9
                )

                response = completion.choices[0].message.content

                # 保存对话历史
                conversation_history.append({"role": "user", "content": user_question})
                conversation_history.append({"role": "assistant", "content": response})

                print(f"\n🎓 AI专家回答:")
                print("-" * 50)
                print(response)
                print("-" * 50)

            except KeyboardInterrupt:
                print("\n\n👋 用户中断，退出咨询系统")
                break
            except Exception as e:
                print(f"\n❌ 处理问题时出现错误: {str(e)}")
                print("💡 请检查网络连接或稍后重试")

    def get_available_cities(self):
        """获取所有可用的城市"""
        cities = self.data['city_name'].unique()
        return sorted(cities)

    def get_available_sectors(self, city):
        """获取指定城市的所有可用部门"""
        city_data = self.data[self.data['city_name'] == city]
        sectors = city_data['sector'].unique()
        return sorted(sectors)


def select_city_and_sector(analyzer):
    """交互式选择城市和部门"""
    print("\n" + "🌍 城市和部门选择".center(60, "="))

    # 1. 显示所有可用城市
    cities = analyzer.get_available_cities()
    print(f"\n📍 可用城市列表 (共{len(cities)}个):")
    print("-" * 40)

    # 按列显示城市，每行4个
    for i in range(0, len(cities), 4):
        row_cities = cities[i:i + 4]
        formatted_cities = [f"{j + i + 1:2d}. {city:<15}" for j, city in enumerate(row_cities)]
        print("  ".join(formatted_cities))

    # 2. 让用户选择城市
    while True:
        try:
            print(f"\n🏙️  请选择城市:")
            print("   - 输入城市编号 (1-{})".format(len(cities)))
            print("   - 或直接输入城市名称")
            print("   - 输入 'quit' 退出程序")

            user_input = input("\n👉 请输入: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                return None, None

            # 尝试作为数字解析
            if user_input.isdigit():
                city_index = int(user_input) - 1
                if 0 <= city_index < len(cities):
                    selected_city = cities[city_index]
                    break
                else:
                    print(f"❌ 请输入1到{len(cities)}之间的数字")
                    continue

            # 尝试作为城市名称匹配
            matching_cities = [city for city in cities if user_input.lower() in city.lower()]
            if len(matching_cities) == 1:
                selected_city = matching_cities[0]
                break
            elif len(matching_cities) > 1:
                print(f"❓ 找到多个匹配的城市: {', '.join(matching_cities)}")
                print("   请输入更具体的名称或使用编号")
                continue
            else:
                print(f"❌ 未找到匹配的城市: {user_input}")
                print("   请检查输入或使用编号选择")
                continue

        except ValueError:
            print("❌ 输入格式错误，请重新输入")
            continue

    print(f"✅ 已选择城市: {selected_city}")

    # 3. 显示该城市的所有部门
    sectors = analyzer.get_available_sectors(selected_city)
    print(f"\n🏢 {selected_city} 可用部门列表 (共{len(sectors)}个):")
    print("-" * 50)

    for i, sector in enumerate(sectors, 1):
        print(f"  {i:2d}. {sector}")

    # 4. 让用户选择部门
    while True:
        try:
            print(f"\n🎯 请选择 {selected_city} 的分析部门:")
            print("   - 输入部门编号 (1-{})".format(len(sectors)))
            print("   - 或直接输入部门名称")
            print("   - 输入 'back' 重新选择城市")
            print("   - 输入 'quit' 退出程序")

            user_input = input("\n👉 请输入: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                return None, None
            elif user_input.lower() in ['back', '返回', 'b']:
                return select_city_and_sector(analyzer)  # 递归调用重新选择

            # 尝试作为数字解析
            if user_input.isdigit():
                sector_index = int(user_input) - 1
                if 0 <= sector_index < len(sectors):
                    selected_sector = sectors[sector_index]
                    break
                else:
                    print(f"❌ 请输入1到{len(sectors)}之间的数字")
                    continue

            # 尝试作为部门名称匹配
            matching_sectors = [sector for sector in sectors if user_input.lower() in sector.lower()]
            if len(matching_sectors) == 1:
                selected_sector = matching_sectors[0]
                break
            elif len(matching_sectors) > 1:
                print(f"❓ 找到多个匹配的部门: {', '.join(matching_sectors)}")
                print("   请输入更具体的名称或使用编号")
                continue
            else:
                print(f"❌ 未找到匹配的部门: {user_input}")
                print("   请检查输入或使用编号选择")
                continue

        except ValueError:
            print("❌ 输入格式错误，请重新输入")
            continue

    print(f"✅ 已选择部门: {selected_sector}")
    print(f"\n🎯 最终选择: {selected_city} - {selected_sector}")

    return selected_city, selected_sector


def main():
    """主函数"""
    print("🌍 碳排放数据AI分析系统启动".center(60, "="))

    try:
        # 初始化分析器
        analyzer = CarbonEmissionAnalyzer(CHINA_DATA, client)

        # 设置分析参数
        # city = 'Shanghai'
        # sector = 'Industry'

        city, sector = select_city_and_sector(analyzer)

        if city is None or sector is None:
            print("👋 用户退出，程序结束")
            return

        print(f"\n📍 分析目标: {city} - {sector}")

        # 1. 数据筛选和可视化
        print("\n📊 第一步: 数据筛选和可视化")
        filtered_data = analyzer.filter_and_plot_data(city, sector)

        if filtered_data is None:
            print("❌ 无法获取数据，程序退出")
            return

        print(f"✅ 成功加载 {len(filtered_data)} 条数据记录")

        # 2. 异常检测
        print("\n🔍 第二步: 异常数据检测")
        anomaly_info = analyzer.detect_anomalies(filtered_data)

        if anomaly_info is None:
            print("❌ 未检测到异常数据，程序退出")
            return

        # 3. 获取新闻背景（可选）
        # print("\n📰 第三步: 新闻背景收集")
        NEWS_API_KEY = '9b99c057d28c405a9d321591f0d2c1c5'  # 请替换为实际的API密钥
        news_headlines = analyzer.get_news_context(
            anomaly_info['city'],
            anomaly_info['date'],
            api_key=None
        )

        # 4. AI深度分析
        print("\n🤖 第三步: AI深度分析")
        analyzer.ai_analysis(anomaly_info, news_headlines)

        # 5. 交互式问答
        print("\n🎯 第四步: 交互式专家咨询")
        while True:
            choice = input("\n❓ 是否需要AI专家进一步咨询？(y/n): ").strip().lower()
            if choice in ['y', 'yes', '是', '1']:
                analyzer.interactive_analysis()
                break
            elif choice in ['n', 'no', '否', '0']:
                print("✨ 分析完成，感谢使用碳排放AI分析系统！")
                break
            else:
                print("❓ 请输入 y 或 n")

    except Exception as e:
        print(f"\n❌ 程序运行出现错误: {str(e)}")
        print("💡 请检查数据文件、网络连接和API配置")


# if __name__ == "__main__":
main()