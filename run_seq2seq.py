from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import numpy as np


input_length = 5
input_dim = 3

output_length = 3
output_dim = 4

samples = 100
hidden_dim = 24


x = np.random.random((samples, input_length, input_dim))
y = np.random.random((samples, output_length, output_dim))

models = []
models += [SimpleSeq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length,
                         input_shape=(input_length, input_dim), depth=2)]

for model in models:
    model.compile(loss='mse', optimizer='sgd')
    model.fit(x, y, nb_epoch=1)