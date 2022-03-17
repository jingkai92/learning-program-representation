import torch
import random
import torch.nn as nn
from factory.embed_factory import NodeEmbedFactory
from utils.pymodels_util import to_cuda, graph_readout, get_reprs
from pymodels.submodels.gnn_layers.gat_layer import GATLayer
from pymodels.submodels.single_vocab.decoder_lstm import DecoderLSTM


class GATSummarizeModel(nn.Module):
    def __init__(self, config):
        super(GATSummarizeModel, self).__init__()
        self.config = config
        self.use_nfeat = self.config.node_emb_layer['use_nfeature']
        self.use_cuda = self.config.use_cuda
        self.graph_config = getattr(self.config, 'gat')
        self.forcing_ratio = 0.75
        self.in_dim = self.graph_config['in_dim']
        self.out_dim = self.graph_config['out_dim']
        self.vocab_len = self.config.token_vocab_dict.vocab_size()

        # Embedding Configurations
        if self.use_nfeat == "structure":
            self.embedding = nn.Embedding(self.vocab_len, self.config.word_emb_dims)
        self.node_emb_layer = NodeEmbedFactory().get_node_embed_technique(self.config)(self.config)

        # GAT Layers
        self.gat_layers = nn.ModuleList([GATLayer(config) for _ in range(self.graph_config['layers'])])

        # Transformation Layers
        self.g_repr = nn.Linear(self.graph_config['out_dim'], self.graph_config['out_dim'])
        self.node_repr = nn.Linear(self.graph_config['out_dim'], self.graph_config['out_dim'])
        self.hid_fc = nn.Linear(self.graph_config['out_dim'], self.graph_config['out_dim'])
        self.cell_fc = nn.Linear(self.graph_config['out_dim'], self.graph_config['out_dim'])

        # Decoder Params
        self.word_emb_dim = self.config.word_emb_dims
        self.lstm_dims = self.config.lstm['dims']
        self.lstm_dropout = self.config.lstm['dropout']

        self.decoder = DecoderLSTM(self.vocab_len,
                                   input_dim=self.config.word_emb_dims,
                                   dec_hid_dim=self.lstm_dims,
                                   use_cuda=self.use_cuda,
                                   bidir=False)

    def forward(self, batch_dict, running_mode, loss_fn):
        g = batch_dict['graphs']
        tgt = to_cuda(batch_dict['tgt_tensors'], use_cuda=self.use_cuda)
        h = to_cuda(g.ndata['node_feat'], self.use_cuda)
        node_len = g.ndata['node_len'].cpu().tolist()
        h = self.node_emb_layer(h, node_len)

        for gat in self.gat_layers:
            h = gat(g, h)
            h = torch.mean(h, dim=1)

        # Due to the head, the size of h is [node_size, nhead, dims]
        # For now, we mean the head output
        g.ndata['h'] = h
        mean_feats = graph_readout(g, self.graph_config['graph_agg'])
        batch_size = mean_feats.shape[0]
        graph_repr, node_reprs, hidden, cell = self.get_representations(g, mean_feats, h)
        src, model_output, loss = self.decoding_phase(running_mode, tgt, batch_size,
                                                      node_reprs, hidden, cell, loss_fn)
        return src, model_output, loss

    def decoding_phase(self, running_mode, target, bsz, node_reprs, hidden, cell, loss_fn):
        if running_mode in ["train", "val"]:
            # tgt = [seq_len, bsz]
            tgt = target.transpose(1, 0)
            tgt_len = tgt.size(0)
        else:
            tgt = None
            tgt_len = self.config.max_sequence_length

        logits, model_output, loss = self.greedy_decode(bsz, tgt_len, hidden,
                                                        cell, node_reprs, tgt=tgt,
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
        # encoder_outputs = encoder_outputs.permute(1, 0, 2)
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

    def get_representations(self, g, mean_feats, h):
        """
        We get the Node Representation from GNNs and
        use linear layer to convert them to different decoder inputs
        :param g: Batched Graph
        :param mean_feats: Aggregated Graph Features
        :param h: Raw Graph Features
        :return:
        """
        return get_reprs(g, mean_feats, h, self.g_repr, self.node_repr,
                         self.hid_fc, self.cell_fc)





