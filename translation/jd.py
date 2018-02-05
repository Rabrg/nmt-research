# coding: utf-8
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torch import optim
# import torch.nn.functional as F

# use_cuda = torch.cuda.is_available()
MAX_LENGTH = 20

# ## Prepare Dataset

# In[2]:


SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# ### List Available Datasets

# In[3]:





# In[30]:


def normalizeEnString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def normalizeCnString(s):
    s = s.strip()
    s = re.sub(r"([\w\u3002\uff1f\uff01])", r" \1", s, flags=re.UNICODE)
    s = re.sub(r"^\s", r"", s)
    return s


def filterPair(pair):
    if (len(pair[0].split(' ')) > MAX_LENGTH - 1
        or len(pair[1].split(' ')) > MAX_LENGTH - 1
        or re.search(r'\w', pair[0])
        or not re.search(r'^((wh)|(i )|(you )|(he )|(she )|(we )|(they ))', pair[1])):
        return False
    return True


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def readFile(fname, pairs):
    f = open(fname, encoding='utf-8')
    lines = f.read().split('\n')
    del lines[-1]
    fPairs = []
    for i in range(0, len(lines), 2):
        if i == len(lines) - 1:
            break
        fPairs.append([normalizeCnString(lines[i + 1]), normalizeEnString(lines[i])])
    print("Read %s sentence pairs from %s" % (len(fPairs), fname))
    fPairs = filterPairs(fPairs)
    print("Trimmed to %s sentence pairs" % len(fPairs))
    pairs.extend(fPairs)


pairs = []
readFile('../UM-Corpus/data/Bilingual/Education/Bi-Education.txt', pairs)
readFile('../UM-Corpus/data/Bilingual/Laws/Bi-Laws.txt', pairs)
readFile('../UM-Corpus/data/Bilingual/Microblog/Bi-Microblog.txt', pairs)
readFile('../UM-Corpus/data/Bilingual/News/Bi-News.txt', pairs)
readFile('../UM-Corpus/data/Bilingual/Science/Bi-Science.txt', pairs)
# readFile('Bi-Spoken.txt', pairs)
readFile('../UM-Corpus/data/Bilingual/Subtitles/Bi-Subtitles.txt', pairs)
readFile('../UM-Corpus/data/Bilingual/Thesis/Bi-Thesis.txt', pairs)

with open('cmn-eng-jd-total.txt', 'w', encoding='utf8') as f:
    for pair in pairs:
        f.write(pair[1] + '\t' + pair[0] + '\n')

        # input_lang = Lang('cn')
        # output_lang = Lang('en')
        #
        # print("\nTotal Sentence Pairs: %d" % len(pairs))
        # print("Counting words...")
        # for pair in pairs:
        #    input_lang.addSentence(pair[0])
        #    output_lang.addSentence(pair[1])
        # print("Counted words:")
        # print(input_lang.name, input_lang.n_words)
        # print(output_lang.name, output_lang.n_words)
        #
        #
        ## ### Randomly Print Several Pairs
        #
        ## In[31]:
        #
        #
        # for i in range(10):
        #    choice = random.choice(pairs)
        #    print(choice[0])
        #    print(choice[1])
        #
        #
        ## ## Encoder
        #
        ## In[32]:
        #
        #
        # class EncoderRNN(nn.Module):
        #    def __init__(self, input_size, hidden_size, n_layers=1):
        #        super(EncoderRNN, self).__init__()
        #        self.n_layers = n_layers
        #        self.hidden_size = hidden_size
        #
        #        self.embedding = nn.Embedding(input_size, hidden_size)
        #        self.gru = nn.GRU(hidden_size, hidden_size)
        #
        #    def forward(self, input, hidden):
        #        embedded = self.embedding(input).view(1, 1, -1)
        #        output = embedded
        #        for i in range(self.n_layers):
        #            output, hidden = self.gru(output, hidden)
        #        return output, hidden
        #
        #    def initHidden(self):
        #        result = Variable(torch.zeros(1, 1, self.hidden_size))
        #        if use_cuda:
        #            return result.cuda()
        #        else:
        #            return result
        #
        #
        ## ## Decoder
        #
        ## In[33]:
        #
        #
        # class AttnDecoderRNN(nn.Module):
        #    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        #        super(AttnDecoderRNN, self).__init__()
        #        self.hidden_size = hidden_size
        #        self.output_size = output_size
        #        self.n_layers = n_layers
        #        self.dropout_p = dropout_p
        #        self.max_length = max_length
        #
        #        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        #        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        #        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        #        self.dropout = nn.Dropout(self.dropout_p)
        #        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        #        self.out = nn.Linear(self.hidden_size, self.output_size)
        #
        #    def forward(self, input, hidden, encoder_output, encoder_outputs):
        #        embedded = self.embedding(input).view(1, 1, -1)
        #        embedded = self.dropout(embedded)
        #
        #        attn_weights = F.softmax(
        #            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        #        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
        #                                 encoder_outputs.unsqueeze(0))
        #
        #        output = torch.cat((embedded[0], attn_applied[0]), 1)
        #        output = self.attn_combine(output).unsqueeze(0)
        #
        #        for i in range(self.n_layers):
        #            output = F.relu(output)
        #            output, hidden = self.gru(output, hidden)
        #
        #        output = F.log_softmax(self.out(output[0]))
        #        return output, hidden, attn_weights
        #
        #    def initHidden(self):
        #        result = Variable(torch.zeros(1, 1, self.hidden_size))
        #        if use_cuda:
        #            return result.cuda()
        #        else:
        #            return result
        #
        #
        ## ## Training
        #
        ## In[34]:
        #
        #
        # def indexesFromSentence(lang, sentence):
        #    return [lang.word2index[word] for word in sentence.split(' ')]
        #
        #
        # def variableFromSentence(lang, sentence):
        #    indexes = indexesFromSentence(lang, sentence)
        #    indexes.append(EOS_token)
        #    result = Variable(torch.LongTensor(indexes).view(-1, 1))
        #    if use_cuda:
        #        return result.cuda()
        #    else:
        #        return result
        #
        #
        # def variablesFromPair(pair):
        #    input_variable = variableFromSentence(input_lang, pair[0])
        #    target_variable = variableFromSentence(output_lang, pair[1])
        #    return (input_variable, target_variable)
        #
        #
        ## In[35]:
        #
        #
        # teacher_forcing_ratio = 0.5
        #
        #
        # def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
        #    encoder_hidden = encoder.initHidden()
        #
        #    encoder_optimizer.zero_grad()
        #    decoder_optimizer.zero_grad()
        #
        #    input_length = input_variable.size()[0]
        #    target_length = target_variable.size()[0]
        #
        #    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        #    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        #
        #    loss = 0
        #
        #    for ei in range(input_length):
        #        encoder_output, encoder_hidden = encoder(
        #            input_variable[ei], encoder_hidden)
        #        encoder_outputs[ei] = encoder_output[0][0]
        #
        #    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        #    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        #
        #    decoder_hidden = encoder_hidden
        #
        #    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        #
        #    if use_teacher_forcing:
        #        # Teacher forcing: Feed the target as the next input
        #        for di in range(target_length):
        #            decoder_output, decoder_hidden, decoder_attention = decoder(
        #                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        #            loss += criterion(decoder_output, target_variable[di])
        #            decoder_input = target_variable[di]  # Teacher forcing
        #
        #    else:
        #        # Without teacher forcing: use its own predictions as the next input
        #        for di in range(target_length):
        #            decoder_output, decoder_hidden, decoder_attention = decoder(
        #                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        #            topv, topi = decoder_output.data.topk(1)
        #            ni = topi[0][0]
        #
        #            decoder_input = Variable(torch.LongTensor([[ni]]))
        #            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        #
        #            loss += criterion(decoder_output, target_variable[di])
        #            if ni == EOS_token:
        #                break
        #
        #    loss.backward()
        #
        #    encoder_optimizer.step()
        #    decoder_optimizer.step()
        #
        #    return loss.data[0] / target_length
        #
        #
        ## In[36]:
        #
        #
        # def asMinutes(s):
        #    m = math.floor(s / 60)
        #    s -= m * 60
        #    return '%dm %ds' % (m, s)
        #
        #
        # def timeSince(since, percent):
        #    now = time.time()
        #    s = now - since
        #    es = s / (percent)
        #    rs = es - s
        #    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
        #
        #
        # def showPlot(points):
        #    plt.figure()
        #    fig, ax = plt.subplots()
        #    # this locator puts ticks at regular intervals
        #    loc = ticker.MultipleLocator(base=0.2)
        #    ax.yaxis.set_major_locator(loc)
        #    plt.plot(points)
        #
        #
        ## In[37]:
        #
        #
        # def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
        #    start = time.time()
        #    plot_losses = []
        #    print_loss_total = 0  # Reset every print_every
        #    plot_loss_total = 0  # Reset every plot_every
        #
        #    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        #    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        #    training_pairs = [variablesFromPair(random.choice(pairs))
        #                      for i in range(n_iters)]
        #    criterion = nn.NLLLoss()
        #
        #    for iter in range(1, n_iters + 1):
        #        training_pair = training_pairs[iter - 1]
        #        input_variable = training_pair[0]
        #        target_variable = training_pair[1]
        #
        #        loss = train(input_variable, target_variable, encoder,
        #                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        #        print_loss_total += loss
        #        plot_loss_total += loss
        #
        #        if iter % print_every == 0:
        #            print_loss_avg = print_loss_total / print_every
        #            print_loss_total = 0
        #            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
        #                                         iter, iter / n_iters * 100, print_loss_avg))
        #
        #        if iter % plot_every == 0:
        #            plot_loss_avg = plot_loss_total / plot_every
        #            plot_losses.append(plot_loss_avg)
        #            plot_loss_total = 0
        #
        #    showPlot(plot_losses)
        #
        #
        ## ## Evaluation
        #
        ## In[38]:
        #
        #
        # def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
        #    input_variable = variableFromSentence(input_lang, sentence)
        #    input_length = input_variable.size()[0]
        #    encoder_hidden = encoder.initHidden()
        #
        #    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        #    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        #
        #    for ei in range(input_length):
        #        encoder_output, encoder_hidden = encoder(input_variable[ei],
        #                                                 encoder_hidden)
        #        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
        #
        #    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
        #    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        #
        #    decoder_hidden = encoder_hidden
        #
        #    decoded_words = []
        #    decoder_attentions = torch.zeros(max_length, max_length)
        #
        #    for di in range(max_length):
        #        decoder_output, decoder_hidden, decoder_attention = decoder(
        #            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        #        decoder_attentions[di] = decoder_attention.data
        #        topv, topi = decoder_output.data.topk(1)
        #        ni = topi[0][0]
        #        if ni == EOS_token:
        #            decoded_words.append('<EOS>')
        #            break
        #        else:
        #            decoded_words.append(output_lang.index2word[ni])
        #
        #        decoder_input = Variable(torch.LongTensor([[ni]]))
        #        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        #
        #    return decoded_words, decoder_attentions[:di + 1]
        #
        #
        ## In[39]:
        #
        #
        # def evaluateRandomly(encoder, decoder, n=10):
        #    for i in range(n):
        #        pair = random.choice(pairs)
        #        print('>', pair[0])
        #        print('=', pair[1])
        #        output_words, attentions = evaluate(encoder, decoder, pair[0])
        #        output_sentence = ' '.join(output_words)
        #        print('<', output_sentence)
        #        print('')
        #
        #
        ## ## Run
        #
        ## In[41]:
        #
        #
        # hidden_size = 256
        # encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
        # attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words)
        #
        # if use_cuda:
        #    encoder1 = encoder1.cuda()
        #    attn_decoder1 = attn_decoder1.cuda()
        #
        # trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
        #
        #
        ## ## Evaluate Some Sentences
        #
        ## In[1]:
        #
        #
        # evaluateRandomly(encoder1, attn_decoder1)
        #
