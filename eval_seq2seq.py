
import torch
import torch.nn as nn
from torch import optim

import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import random
import csv
from util_seq2seq import AttnDecoderRNN, EncoderRNN, Lang

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 24


class EvalSeq2Seq:

    def __init__(self, input_lang: Lang, output_lang: Lang, pairs, encoder: EncoderRNN, attn_decoder: AttnDecoderRNN, device):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs
        self.encoder1 = encoder
        self.attn_decoder1 = attn_decoder
        self.device = device


    def showPlot(self, points, force_show=False, save_figure=False, figure_filename=None):
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)

        if save_figure == True and figure_filename is not None:
            plt.savefig(figure_filename)

        if force_show:
            plt.show()
        else:
            plt.close('all')

    ######################################################################
    # .. note:: There are other forms of attention that work around the length
    #   limitation by using a relative position approach. Read about "local
    #   attention" in `Effective Approaches to Attention-based Neural Machine
    #   Translation <https://arxiv.org/abs/1508.04025>`__.
    #
    # Training
    # ========
    #
    # Preparing Training Data
    # -----------------------
    #
    # To train, for each pair we will need an input tensor (indexes of the
    # words in the input sentence) and target tensor (indexes of the words in
    # the target sentence). While creating these vectors we will append the
    # EOS token to both sequences.
    #

    def indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)

    def tensorsFromPair(self, pair):
        input_tensor = self.tensorFromSentence(self.input_lang, pair[0])
        target_tensor = self.tensorFromSentence(self.output_lang, pair[1])
        return (input_tensor, target_tensor)

    ######################################################################
    # Evaluation
    # ==========
    #
    # Evaluation is mostly the same as training, but there are no targets so
    # we simply feed the decoder's predictions back to itself for each step.
    # Every time it predicts a word we add it to the output string, and if it
    # predicts the EOS token we stop there. We also store the decoder's
    # attention outputs for display later.
    #

    def evaluate(self, encoder: EncoderRNN, decoder: AttnDecoderRNN, sentence, max_length=MAX_LENGTH):
        with torch.no_grad():
            input_tensor = self.tensorFromSentence(self.input_lang, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden(device=self.device)

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=self.device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    w = self.output_lang.index2word[topi.item()]
                    # avoid duplicate needs
                    if w not in decoded_words:
                        decoded_words.append(w)

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]


    ######################################################################
    # We can evaluate random sentences from the training set and print out the
    # input, target, and output to make some subjective quality judgements:
    #

    def evaluateRandomly(self, encoder, decoder, n=4, show_attention=False):
        for i in range(n):
            pair = random.choice(self.pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = self.evaluate(encoder, decoder, pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

            if show_attention == True:
                plt.matshow(attentions.numpy())
                plt.show()

    ######################################################################
    #
    def evaluate_data(self, encoder, decoder, pair_array, output_base: str):
        with open(output_base + '/houston.csv', 'w') as file_pointer:
            translation_writer = csv.writer(file_pointer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            translation_writer.writerow(['Date', 'Hour', 'Minute', 'Wind', 'Pressure', 'Storm Type', 'Category',
                                     'End_hour', "End_minute", "City", "Needs", "translated_needs"])
            for index, pair in enumerate(pair_array):
                print('>', pair[0])
                print('=', pair[1])
                output_words, attentions = self.evaluate(encoder, decoder, pair[0])
                output_sentence = ' '.join(output_words)
                print('<', output_sentence)
                print('')

                s1 = pair[0].split()
                s2 = pair[1]

                translation_writer.writerow(s1 + [s2] + [output_sentence])
                plt.clf()
                # plt.matshow(attentions.numpy())
                filename = output_base + '/attention/att_' + str(index) + ".png"
                self.showAttention(pair[0], output_words, attentions, False, filename)
                plt.close('all')
                break



    ######################################################################
    # Visualizing Attention
    # ---------------------
    #
    # A useful property of the attention mechanism is its highly interpretable
    # outputs. Because it is used to weight specific encoder outputs of the
    # input sequence, we can imagine looking where the network is focused most
    # at each time step.
    #
    # You could simply run ``plt.matshow(attentions)`` to see attention output
    # displayed as a matrix, with the columns being input steps and rows being
    # output steps:
    #

    def do_evaluation(self, encoder1, attn_decoder1):

        output_words, attentions = self.evaluate(
            encoder1, attn_decoder1, "je suis trop froid .")
        plt.matshow(attentions.numpy())

        self.evaluateAndShowAttention("elle a cinq ans de moins que moi .")

        self.evaluateAndShowAttention("elle est trop petit .")

        self.evaluateAndShowAttention("je ne crains pas de mourir .")

        self.evaluateAndShowAttention("c est un jeune directeur plein de talent .")

    ######################################################################
    # For a better viewing experience we will do the extra work of adding axes
    # and labels:
    #

    def showAttention(self, input_sentence, output_words, attentions, force_show=True, filename=None):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_sentence.split(' ') +
                           ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        if force_show == True:
            plt.show()
        else:
            plt.savefig(filename)

    def evaluateAndShowAttention(self, input_sentence):
        output_words, attentions = self.evaluate(
            self.encoder1, self.attn_decoder1, input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        self.showAttention(input_sentence, output_words, attentions)





######################################################################
# Exercises
# =========
#
# -  Try with a different dataset
#
#    -  Another language pair
#    -  Human → Machine (e.g. IOT commands)
#    -  Chat → Response
#    -  Question → Answer
#
# -  Replace the embeddings with pre-trained word embeddings such as word2vec or
#    GloVe
# -  Try with more layers, more hidden units, and more sentences. Compare
#    the training time and results.
# -  If you use a translation file where pairs have two of the same phrase
#    (``I am test \t I am test``), you can use this as an autoencoder. Try
#    this:
#
#    -  Train as an autoencoder
#    -  Save only the Encoder network
#    -  Train a new Decoder for translation from there
#