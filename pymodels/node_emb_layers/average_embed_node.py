import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AverageEmbedNode(nn.Module):
    def __init__(self, config):
        super(AverageEmbedNode, self).__init__()
        self.config = config
        self.word_emb_dims = self.config.word_emb_dims
        self.vocab_len = self.config.token_vocab_dict.vocab_size()
        self.embedding = nn.Embedding(self.vocab_len, self.word_emb_dims, padding_idx=3)

    def forward(self, node_feats, node_lens):
        embed_feats = self.embedding(node_feats)
        embed_feats = torch.mean(embed_feats, dim=1)
        return embed_feats

    def get_embedding_layer(self):
        # Return Token Embedding and Node Type Embedding
        return self.embedding, None


class SingleEmbedNode(nn.Module):
    def __init__(self, config):
        super(SingleEmbedNode, self).__init__()
        self.config = config
        self.word_emb_dims = self.config.word_emb_dims
        self.vocab_len = self.config.token_vocab_dict.vocab_size()
        self.token_embedding = nn.Embedding(self.config.token_vocab_dict.vocab_size(),
                                            self.word_emb_dims, padding_idx=3)
        self.node_embedding = nn.Embedding(self.config.node_vocab_dict.vocab_size(),
                                           self.word_emb_dims, padding_idx=3)

    def forward(self, node_feats, node_lens):
        embed_feats = self.token_embedding(node_feats)

        return embed_feats

    def get_embedding_layer(self):
        # Return Token Embedding and Node Type Embedding
        return self.token_embedding, self.node_embedding
