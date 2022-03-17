import torch.nn as nn


class EmbedNode(nn.Module):
    def __init__(self, config):
        super(EmbedNode, self).__init__()
        self.config = config
        self.node_embedding = nn.Embedding(self.config.node_vocab_dict.vocab_size(),
                                           self.config.word_emb_dims)

    def forward(self, node_feats, node_lens):
        embed_feats = self.node_embedding(node_feats)
        embed_feats = embed_feats.squeeze(1)
        return embed_feats

    def get_embedding_layer(self):
        return None, self.node_embedding
