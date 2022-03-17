import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DistinctEmbedNode(nn.Module):
    def __init__(self, config):
        super(DistinctEmbedNode, self).__init__()
        self.config = config
        self.lstm_dim = self.config.node_emb_layer['dims']
        self.lstm_layer = self.config.node_emb_layer['layers']
        self.word_emb_dims = self.config.word_emb_dims
        self.token_embedding = nn.Embedding(self.config.token_vocab_dict.vocab_size(),
                                            self.word_emb_dims, padding_idx=3)
        self.node_embedding = nn.Embedding(self.config.node_vocab_dict.vocab_size(),
                                           self.word_emb_dims, padding_idx=3)
        self.embed_layer = nn.LSTM(self.lstm_dim,
                                   self.lstm_dim,
                                   num_layers=self.lstm_layer,
                                   bidirectional=False,
                                   batch_first=True)
        self.node_mlp = nn.Linear(self.word_emb_dims, self.word_emb_dims)
        self.lstm_mlp = nn.Linear(self.lstm_dim, self.lstm_dim)
        self.dropout = nn.Dropout(self.config.dropout)
        self.combine_mlp = nn.Linear(self.lstm_dim + self.word_emb_dims, self.lstm_dim)

    def forward(self, node_feats, node_lens):
        node_type = node_feats[:, :1]
        embed_type_feats = self.node_embedding(node_type)
        embed_type_feats = self.node_mlp(embed_type_feats).squeeze()

        # Embed the token part of the node
        node_lens = [x - 1 for x in node_lens]
        node_token = node_feats[:, 1:]
        embed_token_feats = self.token_embedding(node_token)

        last_seq_items = self.get_lstm_feats(embed_token_feats, node_lens)
        mlp_seq = self.lstm_mlp(last_seq_items)

        concat_feats = torch.cat([embed_type_feats, mlp_seq], dim=-1)
        out_feat = self.combine_mlp(concat_feats)
        out_feat = self.dropout(out_feat)
        return out_feat

    def get_lstm_feats(self, embed, lens):
        # We must put the token type into the LSTM
        # Concat them together
        packed_src = pack_padded_sequence(embed, lens,
                                          batch_first=True, enforce_sorted=False)
        outputs, hidtup = self.embed_layer(packed_src)
        outputs, input_sizes = pad_packed_sequence(outputs, batch_first=True)
        last_seq_idxs = torch.tensor([x - 1 for x in lens], dtype=torch.long)
        last_seq_items = outputs[range(outputs.shape[0]), last_seq_idxs, :]
        return last_seq_items

    def get_embedding_layer(self):
        return self.token_embedding, self.node_embedding