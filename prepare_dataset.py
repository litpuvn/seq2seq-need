import collections
import csv
import datetime
import os

def extract_top_words(word_list, count=5):
    counts = collections.Counter(word_list)
    return counts.most_common(count)


data_date = '2017_09_02'
reformatted_data_date = data_date.replace('_', '-')
filepath = 'data/harvey/with_harvey_tag/hourly/cities'


words_by_city = dict()

def get_cities_for_time_block(date, hour, minute):

    city_path = filepath + '/' + date + '/' + (str(hour) + '-' + str(minute))

    if not os.path.exists(city_path):
        return []

    files = [f for f in os.listdir(city_path) if os.path.isfile(os.path.join(city_path, f))]

    cities = [city[0:city.index('_')] for city in files]

    return cities

timestamp_obj = datetime.datetime.strptime(reformatted_data_date + ' 00:00', '%Y-%m-%d %H:%M')
end_time = datetime.datetime.strptime(reformatted_data_date + ' 23:59', '%Y-%m-%d %H:%M')

with open('data/harvey/label/hourly/sample-' + reformatted_data_date + '.csv', 'w') as filePointer:
    csv_writer = csv.writer(filePointer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    row_data = ['Date', 'Start_hour', 'Start_Minute', 'End_Hour', 'End_minute', 'City', 'Needs']
    csv_writer.writerow(row_data)

    while timestamp_obj <= end_time:
        cur_hour = timestamp_obj.strftime('%H')
        cur_minute = timestamp_obj.strftime('%M')

        end_time_block = timestamp_obj + datetime.timedelta(minutes=59, seconds=59)
        end_hour = int(end_time_block.strftime('%H'))
        end_minute = int(end_time_block.strftime('%M'))

        cities = get_cities_for_time_block(reformatted_data_date, end_hour, end_minute)
        for city in cities:
            row_data = [reformatted_data_date, cur_hour, cur_minute, end_hour, end_minute, city, '']
            csv_writer.writerow(row_data)

        timestamp_obj = timestamp_obj + datetime.timedelta(minutes=60)






