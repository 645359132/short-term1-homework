from data import CHINA_DATA


def calculate_average_by_year():
    city_yearly_count = {}
    city_yearly_sum = {}

    for index, row in CHINA_DATA.iterrows():
        city = row["city_name"]
        value = row["value"]
        year = row["year"]

        if city not in city_yearly_sum:
            city_yearly_sum[city] = {}
            city_yearly_count[city] = {}

        if year not in city_yearly_sum[city]:
            city_yearly_sum[city][year] = 0
            city_yearly_count[city][year] = 0

        city_yearly_count[city][year] += 1
        city_yearly_sum[city][year] += value

    city_yearly_avg = {}
    for city in city_yearly_sum:
        city_yearly_avg[city] = {}
        for year in city_yearly_sum[city]:
            city_yearly_avg[city][year] = (
                city_yearly_sum[city][year] / city_yearly_count[city][year]
            )
    print("每个城市每年的平均值：", city_yearly_avg)
    return city_yearly_avg
