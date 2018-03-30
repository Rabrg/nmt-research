import os.path, operator
import random
import nltk
import numpy as np
import torch.nn as nn

from torch import optim
from translation.lang import *
from translation.masked_cross_entropy import *
from translation.time import *
from translation.model import *

import sys
sys.path.append('/home/xing/PycharmProjects/nmt-research-xing-new/translation/')
from model import EncoderRNN
from finetune import fine_tuningSetup

# from translation.model import EncoderRNN, LuongAttnDecoderRNN, LuongAttnDecoderRNNSoftmax
# import translation.model as model

USE_CUDA = True
CUDA_DEVICE = 0
torch.cuda.set_device(CUDA_DEVICE)  # TODO: 0, 1, 2
MULTI_SINGLE = False

PAD_token = 0
SOS_token = 1
EOS_token = 2

MIN_SENT_LENGTH = 1
MAX_SENT_LENGTH = 15  # TODO: 15, 25

# Configure models
attn_model = 'dot'
hidden_size = 200  # TODO: 200, 600
n_layers = 2
dropout = 0.1
# batch_size = 64 REPLACED BY get_batch_size()
networks = 3  # TODO: 1, 3

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_epochs = 50000
epoch = 0
log_every = 100
evaluate_every = 500

def filter_pairs(pairs, input_lang, output_lang):
    keep_pairs = []

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        for word in input_sentence.split(' '):
            if word not in input_lang.word2index:
                keep_input = False
                break

        for word in output_sentence.split(' '):
            if word not in output_lang.word2index:
                keep_output = False
                break

        # Remove if pair doesn't match input and output conditions
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    pairs = keep_pairs
    return pairs


input_lang_total, output_lang_total, pairs_total = prepare_data('de', 'en', 'rapid2016.de-en.de_total.txt', False)
pairs_total = filter_pairs(pairs_total, input_lang_total, output_lang_total)
# input_lang_total, output_lang_total, pairs_total = prepare_data('cmn', 'eng', 'casict2015_total.txt', False)
encoder_path = 'checkpoint_encoder_' + input_lang_total.name + '_' + output_lang_total.name + '.pth.tar'
decoder_path = 'checkpoint_decoder_' + input_lang_total.name + '_' + output_lang_total.name + '.pth.tar'

input_lang_train, output_lang_train, pairs_train = prepare_data('de', 'en', 'rapid2016.de-en.de_train.txt', False)
pairs_train = filter_pairs(pairs_train, input_lang_train, output_lang_train)
# input_lang_train, output_lang_train, pairs_train = prepare_data('cmn', 'eng', 'casict2015_train.txt', False)
input_lang_test, output_lang_test, pairs_test = prepare_data('de', 'en', 'rapid2016.de-en.de_test.txt', False)
pairs_test = filter_pairs(pairs_test, input_lang_test, output_lang_test)
# input_lang_test, output_lang_test, pairs_test = prepare_data('cmn', 'eng', 'casict2015_test.txt', False)


def get_batch_size():
    return 1000


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for _ in range(max_length - len(seq))]
    return seq


def random_batch(batch_size, training_data = True, split=None):
    input_seqs = []
    target_seqs = []

    if training_data:
        # Choose random pairs
        for i in range(batch_size):
            pair = random.choice(pairs_train)  # USES TRAINING SET
            input_seqs.append(indexes_from_sentence(input_lang_total, pair[0]))
            target_seqs.append(indexes_from_sentence(output_lang_total, pair[1]))
    else:
        # Choose random pairs
        for i in range(batch_size):
            pair = random.choice(pairs_test)  # USES TESTING SET
            input_seqs.append(indexes_from_sentence(input_lang_total, pair[0]))
            target_seqs.append(indexes_from_sentence(output_lang_total, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths


def indexed_batch(start, end, batch_size, training_data = True, split=None):
    input_seqs = []
    target_seqs = []

    if training_data:
        # Choose random pairs
        for pair in pairs_train[start:end]:
            input_seqs.append(indexes_from_sentence(input_lang_total, pair[0]))
            target_seqs.append(indexes_from_sentence(output_lang_total, pair[1]))
    else:
        # Choose random pairs
        for pair in pairs_test[start:end]:
            input_seqs.append(indexes_from_sentence(input_lang_total, pair[0]))
            target_seqs.append(indexes_from_sentence(output_lang_total, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths


def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=MAX_SENT_LENGTH):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * get_batch_size()))
    decoder_hidden = encoder_hidden[-decoder.n_layers:]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, get_batch_size(), decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t]  # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0], ec, dc


def load_checkpoint(net, optimizer, epoch, dir, net_index=None):
    if os.path.isfile('../' + dir):
        print("=> loading checkpoint '{}'".format('../' + dir))
        checkpoint = torch.load('../' + dir)
        epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        encoder_embedding.load_state_dict(checkpoint['encoder_embedding'])
        decoder_embedding.load_state_dict(checkpoint['decoder_embedding'])
        # TODO loss_logger[net_index] = checkpoint['loss_logger']
        print("=> loaded checkpoint '{}' (epoch {})".format('../' + dir, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format('../' + dir))
    return net, optimizer, epoch, encoder_embedding, decoder_embedding


# Initialize models
encoder_embedding = nn.Embedding(input_lang_total.n_words, hidden_size)
decoder_embedding = nn.Embedding(output_lang_total.n_words, hidden_size)

encoder = EncoderRNN(encoder_embedding, input_lang_total.n_words, hidden_size, n_layers, dropout=dropout)
use_softmax = False
if not use_softmax:
    decoders = [LuongAttnDecoderRNN(decoder_embedding, attn_model, hidden_size, output_lang_total.n_words, n_layers, dropout=dropout) for i in
                range(networks)]
    criterion = nn.CrossEntropyLoss()
else:
    decoders = [LuongAttnDecoderRNNSoftmax(decoder_embedding, attn_model, hidden_size, output_lang_total.n_words, n_layers, dropout=dropout) for i in
                range(networks)]
    criterion = nn.NLLLoss()
    encoder_path += ".softmax"
    decoder_path += ".softmax"

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate)
decoder_optimizers = [
    optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate * decoder_learning_ratio) for
    decoder in decoders]

# Load previous checkpoints
for i in range(networks):
    encoder_, encoder_optimizer_, ep, enc_emb, dec_emb = load_checkpoint(encoder, encoder_optimizer, epoch,
                                                                       str(i if not MULTI_SINGLE else CUDA_DEVICE) + '_' + str(networks if not MULTI_SINGLE else 3)+ '_' + str(hidden_size) + '_' + encoder_path, i)
    encoder = encoder_
    encoder_optimizer = encoder_optimizer_

    decoder, decoder_optimizer, ep, enc_emb, dec_emb = load_checkpoint(decoders[i], decoder_optimizers[i], epoch,
                                                                       str(i if not MULTI_SINGLE else CUDA_DEVICE) + '_' + str(networks if not MULTI_SINGLE else 3) + '_' + str(hidden_size) + '_' + decoder_path, i)
    decoders[i] = decoder
    decoder_optimizers[i] = decoder_optimizer

    epoch = ep
    encoder_embedding = enc_emb
    decoder_embedding = dec_emb

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    [decoder.cuda() for decoder in decoders]
    encoder_embedding.cuda()
    decoder_embedding.cuda()

loss_logger = [[] for i in range(networks)]
print_loss_total = [0 for i in range(networks)]  # Reset every print_every


def evaluate_ensemble(Ensemble_indice_list, Ensemble_probs_list, max_length=MAX_SENT_LENGTH):
    decoded_words = []
    #Make all lists the same length:
    for List in Ensemble_indice_list:
        while len(List) < max_length:
            List.append(PAD_token)

    for List in Ensemble_probs_list:
        while len(List) < max_length:
            List.append(PAD_token)


    #Bagging
    for di in range(max_length):
        topi_candidate_list = []
        topv_candidate_list = []
        for i in range(len(Ensemble_indice_list)):
            topi = Ensemble_indice_list[i][di]
            topv = Ensemble_probs_list[i][di]
            topi_candidate_list.append(topi)
            topv_candidate_list.append(topv)

        if sum(topi_candidate_list) + sum(topv_candidate_list) == 0:
            ni == EOS_token
            break
        elif len(set(topi_candidate_list)) == networks:
            ni = topi_candidate_list[topv_candidate_list.index(max(topv_candidate_list))]
            # print(output_lang_total.index2word[topi_candidate_list[0]],output_lang_total.index2word[topi_candidate_list[1]],output_lang_total.index2word[topi_candidate_list[2]])
        else:# majority vote
            # Form a dictionary for word counts
            D = {}
            for index in topi_candidate_list:
                try:
                    D[index] += 1
                except KeyError:
                    D[index] = 1
            sorted_D =  sorted(list(D.items()), key=operator.itemgetter(1), reverse=True)
            ni = sorted_D[0][0]
            if ni == PAD_token:
                ni = EOS_token

        if ni == EOS_token:
            break
        else:
            decoded_words.append(output_lang_total.index2word[ni])

    return decoded_words


def evaluate(encoder, decoder, input_seq, max_length=MAX_SENT_LENGTH):
    input_lengths = [len(input_seq)]
    input_seqs = [indexes_from_sentence(input_lang_total, input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)

    if USE_CUDA:
        input_batches = input_batches.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True)  # SOS
    # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    decoder_hidden = encoder_hidden[-decoder.n_layers:]

    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    decoded_words_indices = []
    decoded_words_probs = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    def softmax(array):
        Denominator = sum([np.exp(value) for value in array])
        array = [value / Denominator for value in array]
        return np.asarray(array)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        topProb = softmax(decoder_output.data.cpu().numpy().reshape(output_lang_total.n_words))[ni]
        decoded_words_probs.append(topProb)
        decoded_words_indices.append(ni)


        if ni == EOS_token:
            # decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang_total.index2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))
        if USE_CUDA:
            decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)], decoded_words_indices, decoded_words_probs


def evaluate_not_randomly(encoder, decoder, input_sentence, target_sentence, Ensemble_indice_list, Ensemble_probs_list):
    evaluate_and_show_attention(encoder, decoder, input_sentence, Ensemble_indice_list, Ensemble_probs_list, target_sentence)


def evaluate_randomly(encoder, decoder):
    [input_sentence, target_sentence] = random.choice(pairs_test[networks])  # USES TEST SET
    evaluate_and_show_attention(encoder, decoder, input_sentence, target_sentence)


input_sentence = ''


def evaluate_and_show_ensemble(Ensemble_indice_list, Ensemble_probs_list, target_sentence=None):
    output_words = evaluate_ensemble(Ensemble_indice_list, Ensemble_probs_list)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)


def evaluate_and_show_attention(encoder, decoder, input_sentence, Ensemble_indice_list, Ensemble_probs_list, target_sentence=None):
    output_words, attentions, indice_list, probs_list = evaluate(encoder, decoder, input_sentence)
    Ensemble_indice_list.append(indice_list)
    Ensemble_probs_list.append(probs_list)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)


def save_checkpoint(state, filename):
    torch.save(state, '../' + filename)


while epoch < n_epochs:
    epoch += 1
    network = 0

    input_batches, input_lengths, target_batches, target_lengths = random_batch(get_batch_size())

    for decoder, decoder_optimizer in zip(decoders, decoder_optimizers):
        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        # Keep track of loss
        print_loss_total[network] += loss

        network += 1

    network = 0

    for i in range(networks):
        if epoch % log_every == 0:
            print_loss_avg = print_loss_total[i] / log_every
            loss_logger[i].append(print_loss_avg)
            print_loss_total[i] = 0
            print_summary = 'network %s %s (%d %d%%) %.4f' % (
                i, time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)
            print(loss_logger[i])

    if epoch % evaluate_every == 0:
        network = 0
        [input_sentence, target_sentence] = random.choice(pairs_test)  # USES TEST SET
        Ensemble_indice_list = []
        Ensemble_probs_list = []
        print('sub networks:')
        for decoder, decoder_optimizer in zip(decoders, decoder_optimizers):
            evaluate_not_randomly(encoder, decoder, input_sentence, target_sentence, Ensemble_indice_list, Ensemble_probs_list)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': encoder.state_dict(),
                'optimizer': encoder_optimizer.state_dict(),
                'encoder_embedding': encoder_embedding.state_dict(),
                'decoder_embedding': decoder_embedding.state_dict(),
                'loss_logger': loss_logger[network]
            }, str(network if not MULTI_SINGLE else CUDA_DEVICE) + '_' + str(networks if not MULTI_SINGLE else 3) + '_' + str(hidden_size) + '_' + encoder_path)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': decoder.state_dict(),
                'optimizer': decoder_optimizer.state_dict(),
                'encoder_embedding': encoder_embedding.state_dict(),
                'decoder_embedding': decoder_embedding.state_dict(),
                'loss_logger': loss_logger[network]
            }, str(network if not MULTI_SINGLE else CUDA_DEVICE) + '_' + str(networks if not MULTI_SINGLE else 3) + '_' + str(hidden_size) + '_' + decoder_path)
            network += 1
        print('ensemble:')
        evaluate_and_show_ensemble(Ensemble_indice_list, Ensemble_probs_list)

fine_tuningSetup(random_batch, indexed_batch, pairs_test, encoder, decoders, decoder_embedding, input_lang_total, output_lang_total, False)
