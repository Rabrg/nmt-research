import torch
import nltk
import os
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from masked_cross_entropy import masked_cross_entropy

USE_CUDA = True
PAD_token = 0
SOS_token = 1
EOS_token = 2
networks = 3
clip = 50.0
learning_rate = 0.0001
CUDA_DEVICE = 0
MULTI_SINGLE = False

epoch = 0

def get_batch_size():
    return 1000


def save_checkpoint(state, filename):
    torch.save(state, '../' + filename)


class FineTuningRNN(nn.Module):
    def __init__(self, decoder_embedding, input_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(FineTuningRNN, self).__init__()

        self.decoder_embedding = decoder_embedding
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(200, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seqs, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.decoder_embedding(input_seqs)
        embedded = self.embedding_dropout(embedded)
        # packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(embedded, hidden)
        # outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        outputs = self.out(outputs)
        return outputs


def fineTuning(input_batches, input_lengths, target_batches, target_lengths, encoder, decoders, fRNN, fRNN_optimizer):
    # Set to not-training mode to disable dropout
    encoder.train(False)
    for decoder in decoders:
        decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    max_target_length = max(target_lengths)

    # Moved total_loss initialization outside of the loop
    total_loss = 0
    for decoder in decoders:
        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * get_batch_size()))
        decoder_hidden = encoder_hidden[-decoder.n_layers:]  # Use last (forward) hidden state from encoder
        all_decoder_outputs = Variable(torch.zeros(get_batch_size(), max_target_length))
        # Move new Variables to CUDA
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            all_decoder_outputs[:, t] = decoder_output.data.topk(1)[1]
            decoder_input = target_batches[t]  # Next input is current target

        # all_decoder_outputs is the input of the fRNN
        all_decoder_outputs = all_decoder_outputs.transpose(0, 1).type(torch.LongTensor)

        if USE_CUDA:
            all_decoder_outputs = all_decoder_outputs.cuda()
            fRNN.cuda()

        fRNN_outputs = fRNN(all_decoder_outputs)

        # Loss calculation and backpropagation
        loss = masked_cross_entropy(
            fRNN_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm(fRNN.parameters(), clip)

        # Update parameters with optimizers
        fRNN_optimizer.step()

        total_loss += loss.data[0]

    return total_loss / networks, fRNN_outputs.data.topk(1)[1].squeeze(2).transpose(0, 1)


def load_fRNN_checkpoint(decoder_embedding, net, optimizer, epoch, dir):
    if os.path.isfile('../' + dir):
        print("=> loading checkpoint '{}'".format('../' + dir))
        checkpoint = torch.load('../' + dir)
        epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        decoder_embedding.load_state_dict(checkpoint['decoder_embedding'])
        # TODO loss_logger[net_index] = checkpoint['loss_logger']
        print("=> loaded checkpoint '{}' (epoch {})".format('../' + dir, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format('../' + dir))
    return net, optimizer, epoch, decoder_embedding


def fine_tuning_evaluate(output_lang_total, input_batches, input_lengths, target_batches, target_lengths, encoder, decoders, fRNN_):
    fRNN = fRNN_
    # fRNN_optimizer = fRNN_optimizer_

    # Prepare target words
    ground_truth_indices = target_batches.data.cpu().numpy().transpose()
    ground_truth_sentences = []
    for i in range(get_batch_size()):
        current_sentence = []
        for t in range(ground_truth_indices.shape[1]):
            if ground_truth_indices[i][t] != EOS_token:
                current_sentence.append(output_lang_total.index2word[ground_truth_indices[i][t]])
            else:
                break
        ground_truth_sentences.append(current_sentence)

    # Set to not-training mode to disable dropout
    encoder.train(False)
    fRNN.train(False)
    for decoder in decoders:
        decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    max_target_length = max(target_lengths)

    for decoder in decoders:
        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * get_batch_size()))
        decoder_hidden = encoder_hidden[-decoder.n_layers:]  # Use last (forward) hidden state from encoder
        all_decoder_outputs = Variable(torch.zeros(get_batch_size(), max_target_length))
        # Move new Variables to CUDA
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            all_decoder_outputs[:, t] = decoder_output.data.topk(1)[1]
            decoder_input = target_batches[t]  # Next input is current target

        # all_decoder_outputs is the input of the fRNN
        all_decoder_outputs = all_decoder_outputs.transpose(0, 1).type(torch.LongTensor)

        # Prepare translated words
        translated_indices = all_decoder_outputs.data.cpu().numpy().transpose()
        translated_sentences = []
        for i in range(get_batch_size()):
            current_sentence = []
            for t in range(translated_indices.shape[1]):
                if translated_indices[i][t] != EOS_token:
                    current_sentence.append(output_lang_total.index2word[translated_indices[i][t]])
                else:
                    break
            translated_sentences.append(current_sentence)

        # print("BLEU for subnets:")
        # bleu_score(ground_truth_sentences,translated_sentences)

        if USE_CUDA:
            all_decoder_outputs = all_decoder_outputs.cuda()
            fRNN.cuda()

        fRNN_outputs = fRNN(all_decoder_outputs)
        # Prepare translated words
        translated_indices = fRNN_outputs.data.topk(1)[1].squeeze(2).cpu().numpy().transpose()
        translated_sentences = []
        print(translated_indices.shape)
        for i in range(get_batch_size()):
            current_sentence = []
            for t in range(translated_indices.shape[1]):
                if translated_indices[i][t] != EOS_token:
                    current_sentence.append(output_lang_total.index2word[translated_indices[i][t]])
                else:
                    break
            translated_sentences.append(current_sentence)
        # print("BLEU for ensembled:")
        # bleu_score(ground_truth_sentences,translated_sentences)
        return nltk.translate.bleu_score.corpus_bleu(ground_truth_sentences, translated_sentences,
                                                     smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)


def fine_tuningSetup(random_batch, indexed_batch, pairs_test, encoder, decoders, decoder_embedding, input_lang_total, output_lang_total, dropout, load_fRNN=True):
    # prepare the fine-tuning network
    fRNN = FineTuningRNN(decoder_embedding, input_lang_total.n_words, 600, output_lang_total.n_words, 3,
                         dropout=dropout)

    fRNN_optimizer = optim.Adam(filter(lambda p: p.requires_grad, fRNN.parameters()), lr=learning_rate)

    fRNN_path = 'checkpoint_fRNN_' + input_lang_total.name + '_' + output_lang_total.name + '.pth.tar'

    if load_fRNN:
        # Load up the fRNN
        fRNN, fRNN_optimizer, ep, dec_emb = load_fRNN_checkpoint(decoder_embedding, fRNN, fRNN_optimizer, epoch,
                                                                 str(
                                                                     0 if not MULTI_SINGLE else CUDA_DEVICE) + '_' + str(
                                                                     networks if not MULTI_SINGLE else 3) + '_' + str(
                                                                     600) + '_' + fRNN_path)
    else:
        # Fine-tuning training starts
        print("No pre-trained fRNN found, start training...")
        average_loss = 900.0

        counter = 0

        while average_loss > 1.70:

            # prepare the fine-tuning examples
            input_batches, input_lengths, target_batches, target_lengths = random_batch(get_batch_size())

            average_loss, fRNN_predicted = fineTuning(input_batches, input_lengths, target_batches, target_lengths,
                                                      encoder, decoders, fRNN, fRNN_optimizer)

            counter += 1

            if counter % 100 == 0 and counter != 0:
                print("Average loss is", average_loss)
                cpu_predicted = fRNN_predicted.cpu().numpy()
                ground_truth = target_batches.transpose(0, 1).data.cpu().numpy()
                for i in range(200, 220):
                    ground_truth_sentence = [output_lang_total.index2word[ground_truth[i][t]] for t in
                                             range(ground_truth.shape[1])]
                    ensembled_sentence = [output_lang_total.index2word[cpu_predicted[i][t]] for t in
                                          range(cpu_predicted.shape[1])]
                    print("Truth:", "".join(ground_truth_sentence))
                    print("Ensembled:", "".join(ensembled_sentence))
                    print()

        print("Fine tuning complete after", counter, "iterations, with the final average loss of", average_loss)

        loss_logger = []

        save_checkpoint({
            'epoch': counter + 1,
            'state_dict': fRNN.state_dict(),
            'optimizer': fRNN_optimizer.state_dict(),
            # 'encoder_embedding': encoder_embedding.state_dict(),
            'decoder_embedding': decoder_embedding.state_dict(),
            'loss_logger': loss_logger
        }, str(0 if not MULTI_SINGLE else CUDA_DEVICE) + '_' + str(networks if not MULTI_SINGLE else 3) + '_' + str(
            600) + '_' + fRNN_path)

        print("Checkpoint saved!")

        print("Writing the current batch to files...")

        with open("target_sentences_600.txt", mode='wt', encoding='utf-8') as f:
            for i in range(get_batch_size()):
                ground_truth_sentence = [output_lang_total.index2word[ground_truth[i][t]] for t in
                                         range(ground_truth.shape[1])]

                f.write("".join(ground_truth_sentence) + '\n')

        with open("ensembled_sentences_600.txt", mode='wt', encoding='utf-8') as f:
            for i in range(get_batch_size()):
                ensembled_sentence = [output_lang_total.index2word[cpu_predicted[i][t]] for t in
                                      range(cpu_predicted.shape[1])]
                f.write("".join(ensembled_sentence) + '\n')

        print("Complete!")

    score = 0
    batches = 0

    total_batches = (len(pairs_test) - 1) // get_batch_size()
    for i in range(total_batches):
        input_batches, input_lengths, target_batches, target_lengths = indexed_batch(i * get_batch_size(),
                                                                                     i * get_batch_size() + get_batch_size(),
                                                                                     get_batch_size(),
                                                                                     training_data=False)
        current_score = fine_tuning_evaluate(output_lang_total, input_batches, input_lengths, target_batches, target_lengths, encoder,
                                             decoders, fRNN)
        score += current_score
        batches += 1
        print(score / batches, batches / total_batches)

    print(score / batches)