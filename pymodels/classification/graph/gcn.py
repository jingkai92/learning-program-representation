import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pymodels_util import to_cuda
from factory.embed_factory import NodeEmbedFactory
from pymodels.submodels.gnn_layers.gcn_layer import GCNLayer


class GCNModel(nn.Module):
    def __init__(self, config):
        super(GCNModel, self).__init__()
        self.config = config
        self.use_cuda = self.config.use_cuda
        self.in_dim = self.config.gcn['in_dim']
        self.out_dim = self.config.gcn['out_dim']

        # Embedding Configurations
        self.node_emb_layer = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)

        # GCN Layers
        self.gcn_layers = nn.ModuleList([GCNLayer(config) for _ in range(self.config.gcn['layers'])])

        # Sub Networks
        self.fforward = nn.Linear(self.out_dim, self.config.class_num)

    def forward(self, batch_dict, running_mode, loss_fn):
        g = batch_dict['graphs']
        class_target = batch_dict['tgt']
        h = to_cuda(g.ndata['node_feat'], self.use_cuda)
        node_len = g.ndata['node_len'].cpu().tolist()
        h = self.node_emb_layer(h, node_len)
        for gcn in self.gcn_layers:
            h = gcn(g, h)

        # Remove final feat = h and batch norm
        g.ndata['h'] = h
        mean_feats = dgl.mean_nodes(g, 'h')
        dense_output = F.leaky_relu(self.fforward(mean_feats))

        loss = 0
        if running_mode in ['train', 'val']:
            tgt = to_cuda(torch.tensor(class_target, dtype=torch.long), use_cuda=self.use_cuda)
            loss = loss_fn(dense_output, tgt)
        sm_mask_output = F.softmax(dense_output, dim=-1)
        return sm_mask_output, class_target, loss


