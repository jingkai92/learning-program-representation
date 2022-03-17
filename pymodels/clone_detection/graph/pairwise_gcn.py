import dgl
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from factory.embed_factory import NodeEmbedFactory
from pymodels.submodels.gnn_layers.gcn_layer import GCNLayer
from utils.pymodels_util import to_cuda, graph_readout, get_reprs
from pymodels.submodels.single_vocab.decoder_lstm import DecoderLSTM, DecoderLSTMNoAttention


class PairwiseGCNModel(nn.Module):
    def __init__(self, config):
        super(PairwiseGCNModel, self).__init__()
        # Initialization of Attributes
        self.config = config
        self.use_cuda = self.config.use_cuda
        self.class_num = self.config.class_num
        self.use_nfeat = self.config.node_emb_layer['use_nfeature']
        self.graph_config = getattr(self.config, 'gcn')
        self.in_dim = self.graph_config['in_dim']
        self.out_dim = self.graph_config['out_dim']
        self.vocab_len = self.config.token_vocab_dict.vocab_size()
        self.word_emb_dim = self.config.word_emb_dims

        # Embedding Configurations
        self.node_emb_layer = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)

        # GCN Layers
        self.gcn_layers_one = nn.ModuleList([GCNLayer(config) for _ in range(self.graph_config['layers'])])
        self.gcn_layers_two = nn.ModuleList([GCNLayer(config) for _ in range(self.graph_config['layers'])])
        self.fforward = nn.Linear(self.graph_config['out_dim'], self.config.class_num)

    def forward(self, batch_dict, running_mode, loss_fn):
        class_target = batch_dict['tgt']
        fnone_mean_feats = self.gcn_prop(batch_dict['graphs_one'], self.gcn_layers_one)
        fntwo_mean_feats = self.gcn_prop(batch_dict['graphs_two'], self.gcn_layers_two)

        euc_dist = (fnone_mean_feats - fntwo_mean_feats) ** 2
        dense_output = F.leaky_relu(self.fforward(euc_dist))

        loss = 0
        if running_mode in ['train', 'val']:
            tgt = to_cuda(torch.tensor(class_target, dtype=torch.long),
                          use_cuda=self.use_cuda)
            loss = loss_fn(dense_output, tgt)
        sm_mask_output = F.softmax(dense_output, dim=-1)
        return sm_mask_output, class_target, loss

    def gcn_prop(self, g, gcn_fn):
        h = to_cuda(g.ndata['node_feat'], self.use_cuda)
        node_len = g.ndata['node_len'].cpu().tolist()
        h = self.node_emb_layer(h, node_len)

        for gcn in gcn_fn:
            h = gcn(g, h)

        g.ndata['h'] = h
        mean_feats = graph_readout(g, self.graph_config['graph_agg'])
        return mean_feats
