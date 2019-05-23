# Load Larger LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
import csv



filename = "data/need_dataset.csv"
file_pointer = open(filename)
file_pointer.readline()
raw_text = file_pointer.read()
raw_text = raw_text.replace('\n', ' EOS ')
raw_text = raw_text.replace(',', ' ')
raw_text = raw_text.lower()
raw_text = raw_text.split()
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers

seq_length = 8
dataX = []
# for i in range(0, n_chars - seq_length, 1):
#     seq_in = raw_text[i:i + seq_length]
#     dataX.append([char_to_int[char] for char in seq_in])

with open('data/need_dataset.csv') as file_pointer:
    input_reader = csv.reader(file_pointer, delimiter=',')
    for index, row in enumerate(input_reader):
        if index < 1:
            continue

        city = row[0]
        if city != 'houston':
            continue
        seq_in = row[0:seq_length]
        dataX.append([char_to_int[char.lower()] for char in seq_in])


# one hot encode the output variable
# y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(seq_length, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))

# define the LSTM model
# load the network weights
filename = "data/model/lstm/weights-improvement-20-1.8472.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:", ' '.join([int_to_char[value] for value in pattern]))



need_list = ['boat', 'call', 'charity', 'clothes', 'diaper', 'dog', 'donation', 'food', 'fund', 'gas',
             'help', 'money', 'pet', 'power', 'rescue', 'shelter', 'supplies', 'support', 'text', 'thing',
             'volunteer', 'affect', 'home', 'animal', 'effort', 'relief', 'canoe', 'responder', 'die', 'prayer',
             'governor', 'union', 'corps', 'care', 'trap', 'medical', 'hospital', 'house', 'emergency', 'wish',
             'school', 'life', 'hope', 'family', 'stay', 'hit', 'cat', 'victim', 'recovery', 'shortage',
             'family', 'advisory', 'advice', 'god', 'evacuate', 'friend', 'water', 'update', 'package', 'preparation',
             'repair', 'secure', 'storage', 'sleep', 'supplier', 'survive', 'transportation', 'treatment', 'search', 'ship',
             'castle', 'assist', 'alarm', 'calm', 'aid'
             ]
# generate characters
needs = []
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]

    if result in need_list:
        needs.append(result)
    if result == 'eos' or result == 'EOS':
        print("translated needs:", ' '.join(needs))
        break
    seq_in = [int_to_char[value] for value in pattern]
    # sys.stdout.write(result)

    # print("result: ", result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\nDone.")