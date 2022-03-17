"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl

from factory.embed_factory import NodeEmbedFactory
from pymodels.node_emb_layers.average_embed_node import AverageEmbedNode
from pymodels.node_emb_layers.distinct_embed_node import DistinctEmbedNode
from pymodels.node_emb_layers.embed_node import EmbedNode
from pymodels.node_emb_layers.lstm_embed_node import TextualLSTMEmbedNode
from pymodels.submodels.tree_cell import ChildSumTreeLSTMCell

from utils.pymodels_util import to_cuda, graph_readout
from utils.util import simple_plot_dgl_graph


class TreeLSTMModel(nn.Module):
    def __init__(self, config):
        super(TreeLSTMModel, self).__init__()
        self.config = config
        self.name = "TreeLSTM"
        self.use_cuda = self.config.use_cuda
        # Embedding Configurations
        self.node_emb_layer = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)

        self.dropout = nn.Dropout(self.config.dropout)
        # cell = TreeLSTMCell if self.config.treelstm['cell_type'] == 'nary' else ChildSumTreeLSTMCell
        cell = ChildSumTreeLSTMCell
        self.cell = cell(self.config.word_emb_dims, self.config.treelstm['in_dim'])
        self.fforward = nn.Linear(self.config.treelstm['in_dim'], self.config.class_num)

        self.cur_meanfeats = None

    def forward(self, batch_dict, running_mode, loss_fn):
        class_target = batch_dict['tgt']
        g = batch_dict['graphs']

        n = g.num_nodes()
        mask = to_cuda(g.ndata['mask'], use_cuda=self.config.use_cuda)

        h = to_cuda(th.zeros((n, self.config.treelstm['in_dim'])), use_cuda=self.config.use_cuda)
        c = to_cuda(th.zeros((n, self.config.treelstm['in_dim'])), use_cuda=self.config.use_cuda)

        embeds = self.node_emb_layer(to_cuda(g.ndata['node_feat'], self.use_cuda),
                                     g.ndata['node_len'].cpu().tolist())

        g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds)) * mask.float().unsqueeze(-1)
        g.ndata['h'] = h
        g.ndata['c'] = c

        # propagate
        dgl.prop_nodes_topo(g, self.cell.message_func, self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func)
        # compute logits
        h = self.dropout(g.ndata['h'])
        g.ndata['h'] = h
        mean_feats = F.relu(graph_readout(g, self.config.treelstm['graph_agg']))
        dense_output = F.leaky_relu(self.fforward(mean_feats))
        # graphs = dgl.unbatch(g)
        # root_ids_logits = []
        # for graph in graphs:
        #     root_id = [i for i in range(graph.number_of_nodes()) if graph.out_degrees(i) == 0]
        #     assert len(root_id) == 1, root_id
        #     root_ids_logits.append(graph.ndata['h'][root_id[0]])
        # root_ids_logits_ts = th.stack(root_ids_logits)
        # dense_output = F.leaky_relu(self.fforward(root_ids_logits_ts))
        if running_mode == "test":
            self.cur_meanfeats = mean_feats.cpu()

        loss = 0
        if running_mode in ['train', 'val']:
            tgt = to_cuda(th.tensor(class_target, dtype=th.long),
                          use_cuda=self.use_cuda)
            loss = loss_fn(dense_output, tgt)
        sm_mask_output = F.softmax(dense_output, dim=-1)
        return sm_mask_output, class_target, loss
