import os.path
import random
import nltk
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from torch import optim
from translation.lang import *
from translation.masked_cross_entropy import *
from translation.time import *

USE_CUDA = True
CUDA_DEVICE = 0
torch.cuda.set_device(CUDA_DEVICE)  # TODO: 0, 1, 2
MULTI_SINGLE = False

PAD_token = 0
SOS_token = 1
EOS_token = 2

MIN_SENT_LENGTH = 1
MAX_SENT_LENGTH = 25  # TODO: 15, 25

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
n_epochs = 100000
epoch = 0
log_every = 100
evaluate_every = 500

input_lang_total, output_lang_total, pairs_total = prepare_data('cmn', 'eng', 'casict2015_total.txt', False)
encoder_path = 'checkpoint_encoder_' + input_lang_total.name + '_' + output_lang_total.name + '.pth.tar'
decoder_path = 'checkpoint_decoder_' + input_lang_total.name + '_' + output_lang_total.name + '.pth.tar'

input_lang_train, output_lang_train, pairs_train = prepare_data('cmn', 'eng', 'casict2015_train.txt', False)
input_lang_test, output_lang_test, pairs_test = prepare_data('cmn', 'eng', 'casict2015_test.txt', False)


def get_batch_size():
    return 512


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for _ in range(max_length - len(seq))]
    return seq


def random_batch(batch_size, split=None):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs_train)  # USES TRAINING SET
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


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = encoder_embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(nn.init.xavier_uniform(torch.FloatTensor(1, self.hidden_size)))

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.batch_score(hidden, encoder_outputs)
        return F.softmax(attn_energies).unsqueeze(1)

    def batch_score(self, hidden, encoder_outputs):
        if self.method == 'dot':
            encoder_outputs = encoder_outputs.permute(1, 2, 0)
            energy = torch.bmm(hidden.transpose(0, 1), encoder_outputs).squeeze(1)
        elif self.method == 'general':
            length = encoder_outputs.size(0)
            batch_size = encoder_outputs.size(1)
            energy = self.attn(encoder_outputs.view(-1, self.hidden_size)).view(length, batch_size, self.hidden_size)
            energy = torch.bmm(hidden.transpose(0, 1), energy.permute(1, 2, 0)).squeeze(1)
        elif self.method == 'concat':
            length = encoder_outputs.size(0)
            batch_size = encoder_outputs.size(1)
            attn_input = torch.cat((hidden.repeat(length, 1, 1), encoder_outputs), dim=2)
            energy = self.attn(attn_input.view(-1, 2 * self.hidden_size)).view(length, batch_size, self.hidden_size)
            energy = torch.bmm(self.v.repeat(batch_size, 1, 1), energy.permute(1, 2, 0)).squeeze(1)
        return energy


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = decoder_embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights


class LuongAttnDecoderRNNSoftmax(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNNSoftmax, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = decoder_embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        # output = output.squeeze(0)  # B x N
        # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        output = self.softmax(output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights


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
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

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


def load_checkpoint(net, optimizer, epoch, dir, net_index):
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
encoder = EncoderRNN(input_lang_total.n_words, hidden_size, n_layers, dropout=dropout)
use_softmax = True
if not use_softmax:
    decoders = [LuongAttnDecoderRNN(attn_model, hidden_size, output_lang_total.n_words, n_layers, dropout=dropout) for i in
                range(networks)]
    criterion = nn.CrossEntropyLoss()
else:
    decoders = [LuongAttnDecoderRNNSoftmax(attn_model, hidden_size, output_lang_total.n_words, n_layers, dropout=dropout) for i in
                range(networks)]
    criterion = nn.NLLLoss()
    encoder_path += ".softmax"
    decoder_path += ".softmax"

encoder_embedding = nn.Embedding(input_lang_total.n_words, hidden_size)
decoder_embedding = nn.Embedding(output_lang_total.n_words, hidden_size)

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


def evaluate_ensemble(input_seq, max_length=MAX_SENT_LENGTH):
    # TODO: decoder_outputs is a numpy array and not a FloatTensor. This will run on the CPU instead of GPU,
    # and therefore may hurt performance
    decoder_outputs = np.zeros((max_length, output_lang_total.n_words), dtype=np.float)
    eos_indicies = Counter()

    for decoder in decoders:
        # try:
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
            decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            # Store output words and attention states
            decoded_words = []
            decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

            # Run through decoder
            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

                # Choose top word from output
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                decoder_outputs[di] += decoder_output.data.cpu().numpy().reshape(output_lang_total.n_words)
                if ni == EOS_token:
                    # decoded_words.append('<EOS>')
                    eos_indicies[di] += 1
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
        # except RuntimeError:
        #     print('RuntimeError for input:' + input_seq)

    decoder_outputs /= 3
    decoded_words = []

    for di in range(max_length):
        topi = decoder_outputs[di].argsort()[-1:][::-1]
        ni = topi[0]
        if ni == EOS_token:
            # decoded_words.append('<EOS>')
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
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
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

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]


def evaluate_not_randomly(encoder, decoder, input_sentence, target_sentence):
    evaluate_and_show_attention(encoder, decoder, input_sentence, target_sentence)


def evaluate_randomly(encoder, decoder):
    [input_sentence, target_sentence] = random.choice(pairs_test[networks])  # USES TEST SET
    evaluate_and_show_attention(encoder, decoder, input_sentence, target_sentence)


input_sentence = ''


def evaluate_and_show_ensemble(input_seq, target_sentence=None):
    output_words = evaluate_ensemble(input_seq)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)


def evaluate_and_show_attention(encoder, decoder, input_sentence, target_sentence=None):
    output_words, attentions = evaluate(encoder, decoder, input_sentence)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)


def save_checkpoint(state, filename):
    torch.save(state, '../' + filename)


for i in range(1):
    # Begin!
    while epoch < n_epochs:
        epoch += 1
        network = 0

        input_batches, input_lengths, target_batches, target_lengths = random_batch(get_batch_size(), network)

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
            print('sub networks:')
            for decoder, decoder_optimizer in zip(decoders, decoder_optimizers):
                evaluate_not_randomly(encoder, decoder, input_sentence, target_sentence)
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
            evaluate_and_show_ensemble(input_sentence, target_sentence)

    # total_counter = 0
    # wer_sum = 0
    # hypotheses = []
    # references = []
    # for pair in pairs_test[:20000]:
    #     try:
    #         hypoth = evaluate_ensemble(pair[0])
    #         hypotheses.append(hypoth)
    #     except RuntimeError:
    #         print('Runtime error for pair', pair)
    #         continue
    #     references.append(pair[1])
    #     # print(pair[0] + '\t' + ' '.join(hypoth))
    # print(str(hidden_size), str(networks))
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # print('BLEU SCORE:', nltk.translate.bleu_score.corpus_bleu(references, hypotheses))
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    n_epochs += 10000
