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


class TreeLSTMSummarizeModel(nn.Module):
    def __init__(self, config):
        super(TreeLSTMSummarizeModel, self).__init__()
        self.config = config
        self.forcing_ratio = 0.75
        self.graph_config = getattr(self.config, 'treelstm')
        self.use_cuda = self.config.use_cuda
        self.dropout = nn.Dropout(self.config.dropout)
        self.use_nfeat = self.config.node_emb_layer['use_nfeature']
        self.vocab_len = self.config.token_vocab_dict.vocab_size()
        self.word_emb_dim = self.config.word_emb_dims

        # Embedding Configurations
        if self.use_nfeat == "structure":
            self.embedding = nn.Embedding(self.vocab_len, self.config.word_emb_dims)
        self.node_emb_layer = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)

        # TreeLSTM Layers
        cell = ChildSumTreeLSTMCell
        self.cell = cell(self.word_emb_dim, self.graph_config['in_dim'])

        # Transformation Layers
        self.hid_fc = nn.Linear(self.graph_config['in_dim'], self.config.lstm['dims'])
        self.cell_fc = nn.Linear(self.graph_config['in_dim'], self.config.lstm['dims'])

        # Decoder Params
        self.lstm_dims = self.config.lstm['dims']
        self.lstm_dropout = self.config.lstm['dropout']

        self.decoder = DecoderLSTMNoAttention(self.vocab_len,
                                              input_dim=self.word_emb_dim,
                                              dec_hid_dim=self.lstm_dims,
                                              use_cuda=self.use_cuda,
                                              bidir=False)

    def forward(self, batch_dict, running_mode, loss_fn):
        g = batch_dict['graphs']
        n = g.num_nodes()
        tgt = to_cuda(batch_dict['tgt_tensors'], use_cuda=self.use_cuda)
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
        mean_feats = graph_readout(g, self.graph_config['graph_agg'])
        batch_size = mean_feats.shape[0]

        # Decoding Phase
        if running_mode in ["train", "val"]:
            tgt = tgt.transpose(1, 0)  # tgt = [seq_len, bsz]
            tgt_len = tgt.size(0)
        else:
            tgt = None
            tgt_len = self.config.max_sequence_length

        hidden = self.hid_fc(mean_feats)
        cell = self.cell_fc(mean_feats)
        logits, model_output, loss = self.greedy_decode(batch_size, tgt_len, hidden,
                                                        cell, None, tgt=tgt,
                                                        loss_fn=loss_fn, running_mode=running_mode)
        return "", model_output.transpose(1, 0).tolist(), loss

    def greedy_decode(self, batch_size, tgt_len, hidden, cell, encoder_outputs, tgt=None, loss_fn=None,
                      running_mode=""):
        loss = 0
        if running_mode == "test":
            assert tgt is None
        # Zero is the SOS idx
        input_tensor = to_cuda(torch.zeros(batch_size).long(), self.use_cuda)
        logits_output = to_cuda(torch.zeros(tgt_len, batch_size, self.vocab_len), self.use_cuda)
        model_output = to_cuda(torch.zeros(tgt_len, batch_size), self.use_cuda)
        # [seq_len, bsz, 2* unit] -> [bsz, seq_len, 2* unit]
        token_emb, node_type_emb = self.node_emb_layer.get_embedding_layer()
        # Decoding Part
        for t in range(1, tgt_len):
            # Input Tensor = [bsz]
            if token_emb:
                embed_itenser = token_emb(input_tensor)
            else:
                embed_itenser = self.embedding(input_tensor)
            output, attn, hidden, cell = self.decoder(embed_itenser, hidden, cell, encoder_outputs)
            # output = [batch_size, self.vocab_len]
            logits_output[t] = output
            top1 = output.argmax(1)
            model_output[t] = top1
            if running_mode == "train":
                teacher_force = random.random() < self.forcing_ratio
                input_tensor = tgt[t] if teacher_force else top1
            elif running_mode == "val" or running_mode == "test":
                input_tensor = top1

            if tgt is not None:
                cur_loss = loss_fn(output, tgt[t])
                loss += cur_loss
        return logits_output, model_output, loss
