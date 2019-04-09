import collections
import csv
import os
import datetime


def extract_top_words(word_list, count=5):
    counts = collections.Counter(word_list)
    return counts.most_common(count)


data_date = '2017_08_27'
reformatted_data_date = data_date.replace('_', '-')
filepath = 'data/harvey/with_harvey_tag/clean/harvey_' + data_date + '.csv'

cooccurrence_matrix_path = 'data/harvey/with_harvey_tag/cities'


data_hour = 0
data_start_minute = 0
data_end_minute = data_start_minute + 10


def is_valid_time_block(tweet_date, tweet_hour, tweet_minute):

    global data_hour
    global data_end_minute
    global data_start_minute

    if reformatted_data_date != tweet_date:
        return False

    if data_hour != tweet_hour:
        return False

    if tweet_minute > data_end_minute or tweet_minute < data_start_minute:
        return False

    return True


timestamp_obj = datetime.datetime.strptime(reformatted_data_date + ' 00:00', '%Y-%m-%d %H:%M')
end_time = datetime.datetime.strptime(reformatted_data_date + ' 23:59', '%Y-%m-%d %H:%M')
while timestamp_obj < end_time:
    words_by_city = dict()
    tweets_by_city = dict()
    data_hour = int(timestamp_obj.strftime('%H'))
    data_start_minute = int(timestamp_obj.strftime('%M'))
    data_end_minute = data_start_minute + 9

    with open(filepath) as filepointer:
        text_reader = csv.reader(filepointer, delimiter=',')

        for index, row in enumerate(text_reader):
            if index < 1:
                continue

            t_date = row[1]
            t_hour = int(row[2])
            t_minute = int(row[3])

            ## filter tweets in different time block
            if not is_valid_time_block(tweet_date=t_date, tweet_hour=t_hour, tweet_minute=t_minute):
                continue

            city = row[4].lower()
            text = row[6]

            if city not in words_by_city:
                words_by_city[city] = []
                tweets_by_city[city] = []

            words_by_city[city] = words_by_city[city] + text.split()
            tweets_by_city[city] = tweets_by_city[city] + [text]

        city_top_words = dict()
        city_cooccurrence_matrix = dict()
        for city, words in words_by_city.items():
            top_words = extract_top_words(words)
            top_100_words = extract_top_words(words, count=100)

            city_top_words[city] = top_words

            if city not in city_cooccurrence_matrix:
                city_cooccurrence_matrix[city] = dict()

            # calculate city cooccurrence matrix
            city_tweets = tweets_by_city[city]
            city_matrix = city_cooccurrence_matrix[city]
            for m_tweet in city_tweets:
                for (w1, freq1) in top_words:
                    if w1 not in city_matrix:
                        city_matrix[w1] = dict()
                    w1_city_matrix = city_matrix[w1]
                    for (w2, freq2) in top_100_words:
                        if w2 not in w1_city_matrix:
                            w1_city_matrix[w2] = 0

                        # only increase count if two words in the same tweet
                        if w1 in m_tweet and w2 in m_tweet:
                            w1_city_matrix[w2] = w1_city_matrix[w2] + 1

            print("***** CITY *****", city)

            ## write to file for review
            ## set of co-occurrence metrix of these top lists
            cooccurence_matrix_filename = city + '_' + data_date + '_' + str(data_hour) + '_' + str(data_end_minute) + '.csv'
            parent_dir = cooccurrence_matrix_path + '/' + reformatted_data_date + '/' + (str(data_hour) + '-' + str(data_end_minute))
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            with open(parent_dir + '/' + cooccurence_matrix_filename, 'w') as matrix_writer:
                csv_writer = csv.writer(matrix_writer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                tops = [w for (w, fre) in top_100_words]
                row_data = [''] + tops
                csv_writer.writerow(row_data)
                print(row_data)
                for (w1, freq1) in top_words:
                    row_data = [w1]
                    w1_city_matrix = city_matrix[w1]
                    for (w2, freq2) in top_100_words:
                        w1_w2_count = w1_city_matrix[w2]
                        row_data = row_data + [w1_w2_count]
                    csv_writer.writerow(row_data)
                    print(row_data)


            #print("City", city, ":top words:", top_words)

    timestamp_obj = timestamp_obj + datetime.timedelta(minutes=10)
    print("Size of city", len(city_top_words))


