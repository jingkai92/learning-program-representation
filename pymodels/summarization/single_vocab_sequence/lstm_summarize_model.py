import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from pymodels.submodels.single_vocab.decoder_lstm import DecoderLSTM
from pymodels.submodels.single_vocab.encoder_lstm import EncoderLSTM
from utils.pymodels_util import to_cuda


# Single Vocab Model
class LSTMSummarizeModel(nn.Module):
    def __init__(self, config):
        super(LSTMSummarizeModel, self).__init__()
        self.config = config
        self.use_cuda = self.config.use_cuda
        self.vocab_len = self.config.token_vocab_dict.vocab_size()
        self.batch_size = self.config.batch_size
        self.word_emb_dim = self.config.word_emb_dims
        self.forcing_ratio = 0.75
        self.lstm_dims = self.config.lstm['dims']
        self.lstm_dropout = self.config.lstm['dropout']
        self.lstm_layer = self.config.lstm['layer']
        self.embedding = nn.Embedding(self.vocab_len, self.word_emb_dim, padding_idx=3)

        self.encoder = EncoderLSTM(input_dim=self.word_emb_dim,
                                   hid_dim=self.lstm_dims,
                                   n_layers=self.lstm_layer,
                                   dropout=self.lstm_dropout,
                                   use_cuda=self.use_cuda,
                                   bidir=self.config.lstm['bidirectional'])

        self.decoder = DecoderLSTM(self.vocab_len,
                                   input_dim=self.word_emb_dim,
                                   dec_hid_dim=self.lstm_dims,
                                   use_cuda=self.use_cuda,
                                   bidir=self.config.lstm['bidirectional'])
        self.dropout = nn.Dropout(0.2)
        if self.use_cuda:
            self.embedding = self.embedding.cuda()
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

        embed_src = self.dropout(self.embedding(src))
        packed_src = pack_padded_sequence(embed_src, fn_len, enforce_sorted=False)
        encoder_outputs, hidden, cell = self.encoder(packed_src)

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
        logits_output = torch.zeros(tgt_len, batch_size, self.vocab_len)
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
            embed_itenser = self.embedding(input_tensor)
            output, attn, hidden, cell = self.decoder(embed_itenser, hidden, cell, encoder_outputs)
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
