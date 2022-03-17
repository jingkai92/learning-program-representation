import torch.nn as nn
from dgl.nn.pytorch import GATConv


class GATLayer(nn.Module):
    def __init__(self, config):
        super(GATLayer, self).__init__()
        self.config = config
        self.use_cuda = self.config.use_cuda
        self.in_dim = self.config.gat['in_dim']
        self.out_dim = self.config.gat['out_dim']
        self.nhead = self.config.gat['nhead']
        self.dropout = self.config.gat['dropout']
        self.conv = GATConv(self.in_dim, self.out_dim, self.nhead,
                            self.dropout, self.dropout, allow_zero_in_degree=True)
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
