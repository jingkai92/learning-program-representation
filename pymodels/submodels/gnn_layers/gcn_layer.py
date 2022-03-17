import torch.nn as nn
from dgl.nn.pytorch import GraphConv


class GCNLayer(nn.Module):
    def __init__(self, config):
        super(GCNLayer, self).__init__()
        self.config = config
        self.use_cuda = self.config.use_cuda
        self.in_dim = self.config.gcn['in_dim']
        self.out_dim = self.config.gcn['out_dim']
        self.conv = GraphConv(self.in_dim, self.out_dim, allow_zero_in_degree=True)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.config.dropout)

        if self.use_cuda:
            self.conv = self.conv.cuda()
            self.activation = self.activation.cuda()
            self.dropout = self.dropout.cuda()

    def forward(self, g, h):
        h = self.conv(g, h)
        h = self.activation(h)
        h = self.dropout(h)
        return h
