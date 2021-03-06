# Load Larger LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import os
# load ascii text and covert to lowercase
import csv



filename = "data/fixed_need_dataset.csv"
file_pointer = open(filename)
file_pointer.readline()
raw_text = file_pointer.read()
raw_text = raw_text.replace('\n', ' eos ')
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
dataY = []

with open(filename) as filePointer:
    csv_reader = csv.reader(filePointer, delimiter=',')
    for i, row in enumerate(csv_reader):
        if i < 1:
            continue
        full_row = row[0:(len(row)-1)] + row[len(row)-1].split() + ['eos']
        print('processing..row', i)
        for k in range(0, len(full_row) - seq_length, 1):
            seq_in = full_row[k:(k+seq_length)]
            seq_out = full_row[k + seq_length]
            print('on iteration:', k)
            print('sequence in:', seq_in, "; sequence out:", seq_out)
            dataX.append([char_to_int[char.lower()] for char in seq_in])
            dataY.append(char_to_int[seq_out.lower()])

# for i in range(0, n_chars - seq_length, 1):
#     seq_in = raw_text[i:i + seq_length]
#     seq_out = raw_text[i + seq_length]
#
#     print('on iteration:', i)
#     print('sequence in:', seq_in, "; sequence out:", seq_out)
#     dataX.append([char_to_int[char] for char in seq_in])
#     dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
# model.add(Dropout(0.2))
# model.add(LSTM(256))
model.add(Dropout(0.1))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

base_folder = 'data/model/lstm_' + str(seq_length) + 'seq'
if not os.path.isdir(base_folder):
    os.mkdir(base_folder)

# define the checkpoint
filepath = base_folder + "/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=20, batch_size=8, callbacks=callbacks_list)
