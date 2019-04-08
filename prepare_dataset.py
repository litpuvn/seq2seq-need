import collections
import csv
import datetime

def extract_top_words(word_list, count=5):
    counts = collections.Counter(word_list)
    return counts.most_common(count)


data_date = '2017_08_28'
reformatted_data_date = data_date.replace('_', '-')
filepath = 'data/harvey/with_harvey_tag/clean/harvey_' + data_date + '.csv'


words_by_city = dict()

with open(filepath) as filepointer:
    text_reader = csv.reader(filepointer, delimiter=',')

    for index, row in enumerate(text_reader):
        if index < 1:
            continue

        t_date = row[1]
        t_hour = int(row[2])
        t_minute = int(row[3])


        city = row[4].lower()
        text = row[6]

        if city not in words_by_city:
            words_by_city[city] = []

        words_by_city[city] = words_by_city[city] + text.split()

    timestamp_obj = datetime.datetime.strptime(reformatted_data_date + ' 00:00', '%Y-%m-%d %H:%M')
    end_time = datetime.datetime.strptime(reformatted_data_date + ' 23:59', '%Y-%m-%d %H:%M')

    with open('data/sample-' + reformatted_data_date + '.csv', 'w') as filePointer:
        csv_writer = csv.writer(filePointer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        row_data = ['Date', 'Start_hour', 'Start_Minute', 'End_Hour', 'End_minute', 'City', 'Needs']
        csv_writer.writerow(row_data)

        while timestamp_obj <= end_time:
            cur_hour = timestamp_obj.strftime('%H')
            cur_minute = timestamp_obj.strftime('%M')

            next_time_block = timestamp_obj + datetime.timedelta(minutes=9, seconds=59)
            next_hour = next_time_block.strftime('%H')
            next_minute = next_time_block.strftime('%M')

            for city in words_by_city.keys():
                row_data = [reformatted_data_date, cur_hour, cur_minute, next_hour, next_minute, city, '']
                csv_writer.writerow(row_data)

            timestamp_obj = timestamp_obj + datetime.timedelta(minutes=10)




    print("Done")


