import operator
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tokenizer.vocab_dict import TokenizationConstant
from utils.pymodels_util import convert_logit_to_indices, to_cuda, compute_loss


class PairwiseLSTMModel(nn.Module):
    def __init__(self, config):
        super(PairwiseLSTMModel, self).__init__()
        self.config = config
        self.use_cuda = self.config.use_cuda
        self.vocab_len = self.config.token_vocab_dict.vocab_size()

        self.word_emb_dim = self.config.word_emb_dims
        self.lstm_dims = self.config.lstm['dims']
        self.lstm_dropout = self.config.lstm['dropout']
        self.embedding = nn.Embedding(self.vocab_len, self.word_emb_dim)

        self.encoder_one = nn.LSTM(self.word_emb_dim,
                                   self.lstm_dims,
                                   num_layers=self.config.lstm['layer'],
                                   bidirectional=self.config.lstm['bidirectional'],
                                   dropout=self.lstm_dropout,
                                   batch_first=True)

        self.encoder_two = nn.LSTM(self.word_emb_dim,
                                   self.lstm_dims,
                                   num_layers=self.config.lstm['layer'],
                                   bidirectional=self.config.lstm['bidirectional'],
                                   dropout=self.lstm_dropout,
                                   batch_first=True)
        if self.config.lstm['bidirectional']:
            forward_dims = self.lstm_dims * 2
        else:
            forward_dims = self.lstm_dims

        # Classic Classification
        if hasattr(self.config, "class_num"):
            self.fforward = nn.Linear(forward_dims, self.config.class_num)
        else:
            self.fforward = nn.Linear(forward_dims, self.config.target_vocab_dict.vocab_size())

        if self.use_cuda:
            self.fforward = self.fforward.cuda()
            self.encoder_one = self.encoder_one.cuda()
            self.encoder_two = self.encoder_two.cuda()
            self.embedding = self.embedding.cuda()

    def forward(self, batch_dict, running_mode, loss_fn):
        # src = [bsz, seq_len]
        class_target = batch_dict['tgt']
        # Function one
        packed_fnone_src, fnone_len = self.get_packed_function(batch_dict, 'fn_one_tensors', 'funcs_one_lens')
        packed_fntwo_src, fntwo_len = self.get_packed_function(batch_dict, 'fn_two_tensors', 'funcs_two_lens')

        # Function one LSTM
        fnone_last_items = self.get_last_lstm_output(packed_fnone_src, fnone_len, self.encoder_one)
        fntwo_last_items = self.get_last_lstm_output(packed_fntwo_src, fntwo_len, self.encoder_two)

        euc_dist = (fnone_last_items - fntwo_last_items) ** 2
        dense_output = F.leaky_relu(self.fforward(euc_dist))

        loss = 0
        if running_mode in ['train', 'val']:
            tgt = to_cuda(torch.tensor(class_target, dtype=torch.long), use_cuda=self.use_cuda)
            loss = loss_fn(dense_output, tgt)
        sm_mask_output = F.softmax(dense_output, dim=-1)
        return sm_mask_output, class_target, loss

    def get_packed_function(self, batch_dict, ts_name, len_name):
        fn_tensors = to_cuda(batch_dict[ts_name], use_cuda=self.use_cuda)
        fn_len = batch_dict[len_name]
        embed_fn = self.embedding(fn_tensors)

        # Output = [bsz,  max token num, word emb dim]
        packed_src = pack_padded_sequence(embed_fn, fn_len,
                                          batch_first=True, enforce_sorted=False)
        return packed_src, fn_len

    def get_last_lstm_output(self, packed_src, fn_len, encoder_layer):
        fnone_outputs, (hidden, cell) = encoder_layer(packed_src)
        outputs, input_sizes = pad_packed_sequence(fnone_outputs, batch_first=True)
        last_seq_idxs = torch.tensor([x - 1 for x in fn_len], dtype=torch.long)
        last_seq_items = outputs[range(outputs.shape[0]), last_seq_idxs, :]
        return last_seq_items
