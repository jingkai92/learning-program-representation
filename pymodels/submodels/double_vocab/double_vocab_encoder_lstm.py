import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class DoubleVocabEncoderLSTM(nn.Module):
    def __init__(self, vocab_len, input_dim, hid_dim, n_layers, dropout, use_cuda=True, batch_first=False,
                 bidir=True):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.vocab_len = vocab_len
        self.embedding = nn.Embedding(self.vocab_len, hid_dim, padding_idx=3)

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers,
                           dropout=dropout, bidirectional=bidir,
                           batch_first=batch_first)
        self.bidirectional = bidir
        # Linear Layer to combine final hidden states of LSTM
        fc_hd = hid_dim
        if bidir:
            fc_hd = hid_dim * 2
        self.fc_hidden = nn.Linear(fc_hd, hid_dim)
        self.fc_cell = nn.Linear(fc_hd, hid_dim)

        # For CUDA configuration
        if use_cuda:
            self.rnn = self.rnn.cuda()
            self.fc_hidden = self.fc_hidden.cuda()
            self.fc_cell = self.fc_cell.cuda()

    def forward(self, src, fn_len):
        embed_src = self.embedding(src)
        packed_src = pack_padded_sequence(embed_src, fn_len, enforce_sorted=False)
        # src = [seq_len, bsz, emb_size]
        # outputs = [seq_len, bsz, 2 * hid_dims(for Bidirectional)]
        # hidden, cell = [layers * bidirectional, bsz, hid_dims]
        packed_outputs, (hidden, cell) = self.rnn(packed_src)
        output, input_sizes = pad_packed_sequence(packed_outputs)

        # Concatenate last two items of hidden and cells and put them into a linear layer
        if self.bidirectional:
            hid_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            cell_cat = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)
        else:
            hid_cat = hidden[-1, :, :]
            cell_cat = cell[-1, :, :]
        hfc = self.fc_hidden(hid_cat)
        cfc = self.fc_cell(cell_cat)
        hiddens = torch.tanh(hfc)
        cells = torch.tanh(cfc)
        # hiddens, cells = [bsz, lstm unit]
        # output = [single_vocab_sequence length, bsz, hid_dims]
        return output, hiddens, cells



