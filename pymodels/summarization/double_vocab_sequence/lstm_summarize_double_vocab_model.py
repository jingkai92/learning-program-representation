import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from pymodels.submodels.double_vocab.double_vocab_decoder_lstm import DoubleVocabDecoderLSTM
from pymodels.submodels.double_vocab.double_vocab_encoder_lstm import DoubleVocabEncoderLSTM
from pymodels.submodels.single_vocab.decoder_lstm import DecoderLSTM
from pymodels.submodels.single_vocab.encoder_lstm import EncoderLSTM
from utils.pymodels_util import to_cuda


# Double Vocab Model
class LSTMSummarizeDoubleVocabModel(nn.Module):
    def __init__(self, config):
        super(LSTMSummarizeDoubleVocabModel, self).__init__()
        self.config = config
        self.use_cuda = self.config.use_cuda
        self.src_vocab_len = self.config.token_vocab_dict.vocab_size()
        self.target_vocab_len = self.config.target_vocab_dict.vocab_size()

        self.batch_size = self.config.batch_size
        self.word_emb_dim = self.config.word_emb_dims
        self.forcing_ratio = 0.75
        self.lstm_dims = self.config.lstm['dims']
        self.lstm_dropout = self.config.lstm['dropout']
        self.lstm_layer = self.config.lstm['layer']

        self.encoder = DoubleVocabEncoderLSTM(self.src_vocab_len,
                                              input_dim=self.word_emb_dim,
                                              hid_dim=self.lstm_dims,
                                              n_layers=self.lstm_layer,
                                              dropout=self.lstm_dropout,
                                              use_cuda=self.use_cuda,
                                              bidir=self.config.lstm['bidirectional'])

        self.decoder = DoubleVocabDecoderLSTM(self.target_vocab_len,
                                              input_dim=self.word_emb_dim,
                                              dec_hid_dim=self.lstm_dims,
                                              dropout=self.lstm_dropout,
                                              use_cuda=self.use_cuda,
                                              bidir=self.config.lstm['bidirectional'])

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def forward(self, batch_dict, running_mode, loss_fn):
        # src = [bsz, seq_len]
        src = to_cuda(batch_dict['fn_tensors'], use_cuda=self.use_cuda)
        src = src.transpose(1, 0)
        src_seq_len, batch_size = src.shape
        fn_len = batch_dict['funcs_lens']
        if running_mode in ["train", "val"]:
            tgt = to_cuda(batch_dict['tgt_tensors'], use_cuda=self.use_cuda)
            tgt = tgt.transpose(1, 0)  # tgt = [seq_len, bsz]
            tgt_len = tgt.size(0)
        else:
            tgt = None
            tgt_len = self.config.max_sequence_length

        # Encode the Sequences
        encoder_outputs, hidden, cell = self.encoder(src, fn_len)

        # Decoding Phase
        if running_mode in ["train", "val"]:
            logits, model_output, loss = self.greedy_decode(batch_size, tgt_len, hidden,
                                                            cell, encoder_outputs, tgt=tgt,
                                                            loss_fn=loss_fn, running_mode=running_mode)
        else:
            logits, model_output, loss = self.greedy_decode(batch_size, tgt_len, hidden,
                                                            cell, encoder_outputs, tgt=None,
                                                            loss_fn=loss_fn, running_mode=running_mode)
        return src.transpose(1, 0).tolist(), model_output.transpose(1, 0).tolist(), loss

    def greedy_decode(self, batch_size, tgt_len, hidden, cell, encoder_outputs, tgt=None, loss_fn=None,
                      running_mode=""):
        loss = 0
        # Zero is the SOS idx
        input_tensor = torch.zeros(batch_size).long()
        logits_output = torch.zeros(tgt_len, batch_size, self.target_vocab_len)
        model_output = torch.zeros(tgt_len, batch_size)
        # [seq_len, bsz, 2* unit] -> [bsz, seq_len, 2* unit]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
            logits_output = logits_output.cuda()
            model_output = model_output.cuda()

        # Decoding Part
        for t in range(1, tgt_len):
            # Input Tensor = [bsz]
            output, attn, hidden, cell = self.decoder(input_tensor, hidden, cell, encoder_outputs)
            # output = [batch_size, self.vocab_len]
            logits_output[t] = output
            top1 = output.argmax(1)
            model_output[t] = top1
            if running_mode == "train":
                teacher_force = random.random() < self.forcing_ratio
                input_tensor = tgt[t] if teacher_force else top1
            elif running_mode == "val" or running_mode == "test":
                input_tensor = top1

            if tgt is not None:
                cur_loss = loss_fn(output, tgt[t])
                loss += cur_loss
        return logits_output, model_output, loss
