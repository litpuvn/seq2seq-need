import collections
import csv


def extract_top_words(word_list, count=5):
    counts = collections.Counter(word_list)
    return counts.most_common(count)


data_date = '2017_08_28'
filepath = 'data/harvey/with_harvey_tag/clean/harvey_' + data_date + '.csv'

cooccurrence_matrix_path = 'data/harvey/with_harvey_tag/cities'
words_by_city = dict()
tweets_by_city = dict()

data_hour = 0
data_start_minute = 0
data_end_minute = data_start_minute + 10

def is_valid_time_block(tweet_date, tweet_hour, tweet_minute):

    if data_date != tweet_date:
        return False

    if data_hour != tweet_hour:
        return False

    if tweet_minute > data_end_minute or tweet_minute < data_start_minute:
        return False

    return True

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
            for w1 in top_words:
                if w1 not in city_matrix:
                    city_matrix[w1] = dict()
                w1_city_matrix = city_matrix[w1]
                for w2 in top_100_words:
                    if w2 not in w1_city_matrix:
                        w1_city_matrix[w2] = 0

                    # only increase count if two words in the same tweet
                    if w1 in m_tweet and w2 in m_tweet:
                        w1_city_matrix[w2] = w1_city_matrix[w2] + 1


        ## write to file for review
        ## set of co-occurrence metrix of these top lists
        cooccurence_matrix_filename = city + '_' + data_date + '_' + str(data_hour) + '_' + str(data_end_minute) + '.csv'
        with open(cooccurrence_matrix_path + '', 'w') as matrix_writer:
            csv_writer = csv.writer(matrix_writer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([''] + top_words)
            for w1 in top_words:
                row_data = [w1]
                for w2 in top_100_words:
                    w1_w2_count = w1_city_matrix[w2]
                    row_data = row_data + [w1_w2_count]
                csv_writer.writerow(row_data)


        print("done")
        #print("City", city, ":top words:", top_words)

    print("Size of city", len(city_top_words))


