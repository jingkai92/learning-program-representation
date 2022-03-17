"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import random
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl

from factory.embed_factory import NodeEmbedFactory
from pymodels.node_emb_layers.average_embed_node import AverageEmbedNode
from pymodels.node_emb_layers.distinct_embed_node import DistinctEmbedNode
from pymodels.node_emb_layers.embed_node import EmbedNode
from pymodels.node_emb_layers.lstm_embed_node import TextualLSTMEmbedNode
from pymodels.submodels.single_vocab.decoder_lstm import DecoderLSTM, DecoderLSTMNoAttention
from pymodels.submodels.tree_cell import ChildSumTreeLSTMCell
from utils.pymodels_util import to_cuda, graph_readout


class PairwiseTreeLSTMModel(nn.Module):
    def __init__(self, config):
        super(PairwiseTreeLSTMModel, self).__init__()
        self.config = config
        self.forcing_ratio = 0.75
        self.graph_config = getattr(self.config, 'treelstm')
        self.use_cuda = self.config.use_cuda
        self.dropout = nn.Dropout(self.config.dropout)
        self.use_nfeat = self.config.node_emb_layer['use_nfeature']
        self.vocab_len = self.config.token_vocab_dict.vocab_size()
        self.word_emb_dim = self.config.word_emb_dims

        # Embedding Configurations
        self.node_emb_layer = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)

        # TreeLSTM Layers
        cell = ChildSumTreeLSTMCell
        self.cell_one = cell(self.word_emb_dim, self.graph_config['in_dim'])
        self.cell_two = cell(self.word_emb_dim, self.graph_config['in_dim'])
        self.fforward = nn.Linear(self.config.treelstm['in_dim'], self.config.class_num)

    def forward(self, batch_dict, running_mode, loss_fn):
        class_target = batch_dict['tgt']
        fnone_mean_feats = self.treelstm_layer(batch_dict['graphs_one'], self.cell_one)
        fntwo_mean_feats = self.treelstm_layer(batch_dict['graphs_two'], self.cell_two)

        euc_dist = (fnone_mean_feats - fntwo_mean_feats) ** 2
        dense_output = F.leaky_relu(self.fforward(euc_dist))

        loss = 0
        if running_mode in ['train', 'val']:
            tgt = to_cuda(th.tensor(class_target, dtype=th.long),
                          use_cuda=self.use_cuda)
            loss = loss_fn(dense_output, tgt)
        sm_mask_output = F.softmax(dense_output, dim=-1)
        return sm_mask_output, class_target, loss

    def treelstm_layer(self, g, cell_layer):
        n = g.num_nodes()
        mask = to_cuda(g.ndata['mask'], use_cuda=self.config.use_cuda)

        h = to_cuda(th.zeros((n, self.config.treelstm['in_dim'])), use_cuda=self.config.use_cuda)
        c = to_cuda(th.zeros((n, self.config.treelstm['in_dim'])), use_cuda=self.config.use_cuda)

        # We always use the same embedding layer
        embeds = self.node_emb_layer(to_cuda(g.ndata['node_feat'], self.use_cuda),
                                     g.ndata['node_len'].cpu().tolist())

        # Different TreeLSTM for each graph
        g.ndata['iou'] = cell_layer.W_iou(self.dropout(embeds)) * mask.float().unsqueeze(-1)
        g.ndata['h'] = h
        g.ndata['c'] = c

        # propagate
        dgl.prop_nodes_topo(g, cell_layer.message_func, cell_layer.reduce_func,
                            apply_node_func=cell_layer.apply_node_func)

        # compute logits
        h = self.dropout(g.ndata['h'])
        g.ndata['h'] = h
        mean_feats = graph_readout(g, self.graph_config['graph_agg'])
        return mean_feats
