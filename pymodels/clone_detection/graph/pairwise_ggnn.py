import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GatedGraphConv

from factory.embed_factory import NodeEmbedFactory
from pymodels.node_emb_layers.average_embed_node import AverageEmbedNode
from pymodels.node_emb_layers.distinct_embed_node import DistinctEmbedNode
from pymodels.node_emb_layers.embed_node import EmbedNode
from pymodels.node_emb_layers.lstm_embed_node import TextualLSTMEmbedNode, BothLSTMEmbedNode
from utils.pymodels_util import to_cuda, graph_readout


class PairwiseGGNNModel(nn.Module):
    def __init__(self, config):
        super(PairwiseGGNNModel, self).__init__()
        self.config = config
        self.use_cuda = self.config.use_cuda
        self.class_num = self.config.class_num
        self.use_nfeat = self.config.node_emb_layer['use_nfeature']
        self.graph_config = getattr(self.config, 'ggnn')
        self.in_dim = self.graph_config['in_dim']
        self.out_dim = self.graph_config['out_dim']
        self.nsteps = self.graph_config['nsteps']
        self.vocab_len = self.config.token_vocab_dict.vocab_size()
        self.word_emb_dim = self.config.word_emb_dims

        # Embedding Configurations
        self.node_emb_layer = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)

        # Graph Configuration
        self.fnone_g_conv = GatedGraphConv(self.in_dim, self.out_dim, self.nsteps, len(self.config.edge_type_list))
        self.fntwo_g_conv = GatedGraphConv(self.in_dim, self.out_dim, self.nsteps, len(self.config.edge_type_list))

        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.config.dropout)
        # Sub Networks
        fforward_dims = self.out_dim
        if self.config.ggnn['initial_representation']:
            fforward_dims = self.out_dim * 2
        self.batch_norm_one = nn.BatchNorm1d(fforward_dims)
        self.batch_norm_two = nn.BatchNorm1d(fforward_dims)
        self.fforward = nn.Linear(fforward_dims, self.config.class_num)

        if self.use_cuda:
            self.fnone_g_conv = self.fnone_g_conv.cuda()
            self.fntwo_g_conv = self.fntwo_g_conv.cuda()
            self.batch_norm_one = self.batch_norm_one.cuda()
            self.batch_norm_two = self.batch_norm_two.cuda()
            self.activation = self.activation.cuda()
            self.dropout = self.dropout.cuda()
            self.fforward = self.fforward.cuda()

    def forward(self, batch_dict, running_mode, loss_fn):
        class_target = batch_dict['tgt']
        fnone_mean_feats = self.get_ggnn_feats(batch_dict['graphs_one'], self.fnone_g_conv, self.batch_norm_one)
        fntwo_mean_feats = self.get_ggnn_feats(batch_dict['graphs_two'], self.fntwo_g_conv, self.batch_norm_two)
        euc_dist = (fnone_mean_feats - fntwo_mean_feats) ** 2
        dense_output = F.leaky_relu(self.fforward(euc_dist))

        loss = 0
        if running_mode in ['train', 'val']:
            tgt = to_cuda(torch.tensor(class_target, dtype=torch.long),
                          use_cuda=self.use_cuda)
            loss = loss_fn(dense_output, tgt)
        sm_mask_output = F.softmax(dense_output, dim=-1)
        return sm_mask_output, class_target, loss

    def get_ggnn_feats(self, g, g_layer, batch_norm_layer):
        h = to_cuda(g.ndata['node_feat'], self.use_cuda)
        node_len = g.ndata['node_len'].cpu().tolist()
        elist = to_cuda(g.edata['edge_type'], self.use_cuda)
        embed_h = self.node_emb_layer(h, node_len)

        h = g_layer(g, embed_h, elist)
        if self.graph_config['initial_representation']:
            h = torch.cat([h, embed_h], -1)
        h = batch_norm_layer(h)
        h = self.dropout(h)
        g.ndata['h'] = h
        mean_feats = graph_readout(g, self.graph_config['graph_agg'])
        return mean_feats