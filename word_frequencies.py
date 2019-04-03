import collections
import csv


def top_words(word_list, count=100):
    counts = collections.Counter(word_list)
    return counts.most_common(count)


filepath = 'data/harvey/harvey_2017_08_17.csv'
tweet_by_city = dict()
with open(filepath) as filepointer:
    text_reader = csv.reader(filepointer, delimiter=',')

    for index, row in enumerate(text_reader):
        if index < 1:
            continue

        tweet_date = row[1]
        tweet_hour = row[2]
        tweet_minute = row[3]

        city = row[4].lower()
        text = row[6]

        if city not in tweet_by_city:
            tweet_by_city[city] = []

        word_list = tweet_by_city[city]
        word_list = word_list + text.split()

    city_top_words = dict()
    for city, words in tweet_by_city.items():
        top_words = top_words(words)
        city_top_words[city] = top_words

        print("City", city, ":top words:", top_words)


