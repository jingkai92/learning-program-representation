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


class GGNNModel(nn.Module):
    def __init__(self, config):
        super(GGNNModel, self).__init__()
        self.config = config
        self.name = "ggnn"
        self.cur_meanfeats = None
        self.use_cuda = self.config.use_cuda
        self.class_num = self.config.class_num
        self.graph_config = getattr(self.config, 'ggnn')
        self.use_nfeat = self.config.node_emb_layer['use_nfeature']
        self.in_dim = self.graph_config['in_dim']
        self.out_dim = self.graph_config['out_dim']
        self.nsteps = self.graph_config['nsteps']

        # Embedding Configuration
        self.node_emb_layer = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)

        # Graph Convolution
        self.g_conv = GatedGraphConv(self.in_dim, self.out_dim, self.nsteps, len(self.config.edge_type_list))

        # Activation and Batch Norms
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.config.dropout)

        # Sub Networks
        fforward_dims = self.out_dim
        if self.config.ggnn['initial_representation']:
            fforward_dims = self.out_dim * 2
        self.batch_norm = nn.BatchNorm1d(fforward_dims)
        self.fforward = nn.Linear(fforward_dims, self.class_num)

    def forward(self, batch_dict, running_mode, loss_fn):
        g = batch_dict['graphs']
        class_target = batch_dict['tgt']
        h = to_cuda(g.ndata['node_feat'], self.use_cuda)
        node_len = g.ndata['node_len'].cpu().tolist()
        elist = to_cuda(g.edata['edge_type'], self.use_cuda)
        embed_h = self.node_emb_layer(h, node_len)

        h = F.leaky_relu(self.g_conv(g, embed_h, elist))
        if self.config.ggnn['initial_representation']:
            h = torch.cat([h, embed_h], -1)
        h = self.batch_norm(h)
        h = self.dropout(h)
        g.ndata['h'] = h
        mean_feats = graph_readout(g, self.graph_config['graph_agg'])
        if running_mode == "test":
            self.cur_meanfeats = mean_feats.cpu()
        dense_output = F.leaky_relu(self.fforward(mean_feats))
        loss = 0
        if running_mode in ['train', 'val']:
            tgt = to_cuda(torch.tensor(class_target, dtype=torch.long),
                          use_cuda=self.use_cuda)
            loss = loss_fn(dense_output, tgt)
        sm_mask_output = F.softmax(dense_output, dim=-1)
        return sm_mask_output, class_target, loss


