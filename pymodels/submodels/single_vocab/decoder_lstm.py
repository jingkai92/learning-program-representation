import torch
import torch.nn as nn
from pymodels.submodels.attention import Attention


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_len, input_dim, dec_hid_dim, use_cuda=True,
                 gnn_dims=-1, bidir=True):
        super().__init__()
        self.vocab_len = vocab_len
        self.dec_hid_dim = dec_hid_dim
        self.use_cuda = use_cuda

        # With attention, LSTM input dimension is 2 * Encoder Dimension + Embedding Dimension
        # 2 * Encoder Dimension is for Bidirectional LSTM, while Embedding Dimension = self.config.word_emb_dim
        if bidir:
            lstm_input_dim = (2 * dec_hid_dim) + input_dim
            attn_input_dim = (2 * dec_hid_dim) + dec_hid_dim
        else:
            lstm_input_dim = dec_hid_dim + input_dim
            attn_input_dim = dec_hid_dim + dec_hid_dim

        lstm_hd = dec_hid_dim
        if gnn_dims != -1:
            attn_input_dim = attn_input_dim + gnn_dims
            lstm_hd = dec_hid_dim + gnn_dims

        self.rnn = nn.LSTM(lstm_input_dim, lstm_hd, 1, dropout=0)
        self.attention = Attention(attn_input_dim, dec_hid_dim)
        self.fc_out = nn.Linear(lstm_hd, vocab_len)
        if self.use_cuda:
            self.rnn = self.rnn.cuda()
            self.attention = self.attention.cuda()
            self.fc_out = self.fc_out.cuda()

    def forward(self, input_seq, hidden, cell, encoder_outputs):
        embedded = input_seq.unsqueeze(0)  # Current Decoder Input
        rnn_input, attn = self._apply_attn(embedded, hidden, encoder_outputs)
        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, attn, hidden.squeeze(0), cell.squeeze(0)

    def _apply_attn(self, embedded, hidden, encoder_outputs):
        # Compute attention vector A
        attn = self.attention(hidden, encoder_outputs)
        attn = attn.unsqueeze(1)
        weighted = torch.bmm(attn, encoder_outputs)
        # This should be the attention vector
        weighted = weighted.permute(1, 0, 2)
        # Embedded, Weighted Encoder Outputs
        # rnn_input = [1, bsz, 2 * unit + emb size]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        return rnn_input, attn


class DecoderLSTMNoAttention(nn.Module):
    def __init__(self, vocab_len, input_dim, dec_hid_dim, use_cuda=True,
                 gnn_dims=-1, bidir=True):
        super().__init__()
        self.vocab_len = vocab_len
        self.dec_hid_dim = dec_hid_dim
        self.use_cuda = use_cuda

        # With attention, LSTM input dimension is 2 * Encoder Dimension + Embedding Dimension
        # 2 * Encoder Dimension is for Bidirectional LSTM, while Embedding Dimension = self.config.word_emb_dim
        if bidir:
            lstm_input_dim = (2 * dec_hid_dim)
        else:
            lstm_input_dim = dec_hid_dim

        lstm_hd = dec_hid_dim
        if gnn_dims != -1:
            lstm_hd = dec_hid_dim + gnn_dims

        self.rnn = nn.LSTM(lstm_input_dim, lstm_hd, 1, dropout=0)
        self.fc_out = nn.Linear(lstm_hd, vocab_len)
        if self.use_cuda:
            self.rnn = self.rnn.cuda()
            self.fc_out = self.fc_out.cuda()

    def forward(self, input_seq, hidden, cell, encoder_outputs):
        embedded = input_seq.unsqueeze(0)  # Current Decoder Input
        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, None, hidden.squeeze(0), cell.squeeze(0)
