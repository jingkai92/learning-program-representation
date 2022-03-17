import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextualLSTMEmbedNode(nn.Module):
    def __init__(self, config):
        super(TextualLSTMEmbedNode, self).__init__()
        self.config = config
        self.lstm_dim = self.config.node_emb_layer['dims']
        self.lstm_layer = self.config.node_emb_layer['layers']
        self.word_emb_dims = self.config.word_emb_dims
        self.vocab_len = self.config.token_vocab_dict.vocab_size()
        self.embedding = nn.Embedding(self.vocab_len, self.word_emb_dims)
        self.embed_layer = nn.LSTM(self.lstm_dim,
                                   self.lstm_dim,
                                   num_layers=self.lstm_layer,
                                   bidirectional=False,
                                   batch_first=True)

    def forward(self, node_feats, node_lens):
        embed_feats = self.embedding(node_feats)
        packed_src = pack_padded_sequence(embed_feats, node_lens,
                                          batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.embed_layer(packed_src)
        outputs, input_sizes = pad_packed_sequence(outputs, batch_first=True)
        last_seq_idxs = torch.tensor([x - 1 for x in node_lens], dtype=torch.long)
        last_seq_items = outputs[range(outputs.shape[0]), last_seq_idxs, :]
        return last_seq_items

    def get_embedding_layer(self):
        # Return Token Embedding and Node Type Embedding
        return self.embedding, None


class BothLSTMEmbedNode(nn.Module):
    def __init__(self, config):
        super(BothLSTMEmbedNode, self).__init__()
        self.config = config
        self.lstm_dim = self.config.node_emb_layer['dims']
        self.lstm_layer = self.config.node_emb_layer['layers']
        self.word_emb_dims = self.config.word_emb_dims
        self.token_embedding = nn.Embedding(self.config.token_vocab_dict.vocab_size(), self.word_emb_dims)
        self.node_embedding = nn.Embedding(self.config.node_vocab_dict.vocab_size(), self.word_emb_dims)
        self.embed_layer = nn.LSTM(self.lstm_dim,
                                   self.lstm_dim,
                                   num_layers=self.lstm_layer,
                                   bidirectional=False,
                                   batch_first=True)
        self.node_mlp = nn.Linear(self.word_emb_dims, self.word_emb_dims)
        self.lstm_mlp = nn.Linear(self.lstm_dim, self.lstm_dim)

    def forward(self, node_feats, node_lens):
        node_type = node_feats[:, :1]
        embed_type_feats = self.node_embedding(node_type)
        embed_type_feats = self.node_mlp(embed_type_feats)

        # Embed the token part of the node
        node_lens = [x - 1 for x in node_lens]
        node_token = node_feats[:, 1:]
        embed_token_feats = self.token_embedding(node_token)

        # We must put the token type into the LSTM
        # Concat them together
        embed_feats = torch.cat([embed_type_feats, embed_token_feats], dim=1)
        packed_src = pack_padded_sequence(embed_feats, node_lens,
                                          batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.embed_layer(packed_src)
        outputs, input_sizes = pad_packed_sequence(outputs, batch_first=True)
        last_seq_idxs = torch.tensor([x - 1 for x in node_lens], dtype=torch.long)
        last_seq_items = outputs[range(outputs.shape[0]), last_seq_idxs, :]
        mlp_seq = self.lstm_mlp(last_seq_items)
        return mlp_seq

    def get_embedding_layer(self):
        # Return Token Embedding and Node Type Embedding
        return self.token_embedding, self.node_embedding