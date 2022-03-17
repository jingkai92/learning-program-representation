import dgl
import torch
from bases.base_graph_formatter import BaseGraphFormatter
from utils.pymodels_util import pad_to_max, stack_seq_to_tensor


class SingleEdgeGraphFormatter(BaseGraphFormatter):
    def __init__(self, config, name="SingleEdgeGraphFormatter"):
        """
        SingleEdgeGraphFormatter will format the input data.
        """
        self.name = name
        self.config = config
        BaseGraphFormatter.__init__(self, config, name)

    def format(self, item_json, vocab_dicts):
        """
        Format single item_json using the Vocab Dictionary
        {'IS_AST_PARENT': 0, 'isCFGParent': 1, "POST_DOM": 2,
        "FLOWS_TO": 3, "USE": 4, "DEF": 5, 'REACHES': 6, "CONTROLS": 7}
        :param item_json: JSON of a single item in the dataset
        :param vocab_dicts:  ["Token", "Node", "Target"].
        :return: Return Datapoints
        """
        token_vd, node_vd, target_vd, word_vd = vocab_dicts
        datapoint = self.datapoint_class()
        dgl_graph = self._convert_to_dglgraph(item_json['jsgraph'], token_vd,
                                              node_vd)
        datapoint.function_graph = dgl_graph
        datapoint.function = item_json['function']
        datapoint.graph_size = item_json['graph_size']
        self._set_target(datapoint, item_json['target'], token_vd)
        return datapoint

    def _set_target(self, dp, target, vd):
        """
        Two types of Target, String and Int
        :param dp: Datapoint
        :param target: Target
        :param vd: Vocab Dictionary
        :return: NIL
        """
        if type(target) == int:
            dp.tgt = target
        else:
            tok_tgt = self.t3_parser.tokenize(target)
            dp.tgt = tok_tgt
            tok_tgt, blen = self.tokenize_sentence(tok_tgt, vd)
            dp.tgt_vec = tok_tgt

    def _convert_to_dglgraph(self, jsgraph, token_vd, node_vd):
        """
        Convert the JSGraph to DGL Graph
        Since it is Isotropic Graph, edge type is not considered.
        We will map all edges into a single type - 0
        :param jsgraph: JSGraph
        :param token_vd: Token Vocab Dictionary
        :param node_vd: Node Vocab Dictionary
        :return: Return a DGL Graph
        """
        edges = jsgraph['graph']
        node_features = jsgraph['node_features']
        src_edges = [x[0] for x in edges]
        dst_edges = [x[1] for x in edges]
        # Each element i is map to each element k in src and dst
        g = dgl.graph((src_edges, dst_edges))
        assert g.num_nodes() == len(node_features)

        node_feat_vecs = [0] * g.num_nodes()
        node_feat_lens = [0] * g.num_nodes()
        for key, node_feat in node_features.items():
            node_feat, node_feat_len = self.get_feats(node_feat, token_vd, node_vd)
            node_feat_vecs[int(key)] = node_feat
            node_feat_lens[int(key)] = node_feat_len
        g.ndata['node_feat'] = stack_seq_to_tensor(node_feat_vecs)
        g.ndata['node_len'] = torch.tensor(node_feat_lens, dtype=torch.long)
        if self.config.use_cuda:
            g = g.to('cuda')
        return g

    def get_feats(self, node_feat, token_vd, node_vd):
        """
        Get the respective features based on either structure or textual
        Sample Node Feature: ['Function', '', 0, False]
        :param node_feat: The four elements of Node Item
        :param token_vd: Token Vocab Dictionary
        :param node_vd: Node Vocab Dictionary
        :return: Return a vectorized vec
        """
        use_feats = self.config.node_emb_layer['use_nfeature']
        if use_feats == "structure":
            struct_vec, struct_len = self.get_structure_feat(node_feat, node_vd)
            return struct_vec, struct_len
        elif use_feats == "textual":
            text_vec, text_len = self.get_textual_feat(node_feat, token_vd)
            return text_vec, text_len
        elif use_feats == "both":
            struct_vec, struct_len = self.get_structure_feat(node_feat, node_vd)
            text_vec, text_len = self.get_textual_feat(node_feat, token_vd)
            assert len(struct_vec) == 1
            text_vec.insert(0, struct_vec[0])
            return text_vec, text_len + 1

    def get_structure_feat(self, node_feat, node_vd):
        """
        Sample Node Feature: ['Function', '', 0, False]
        :param node_feat: The four elements of Node Item
        :param node_vd: Node Vocab Dictionary
        :return:
        """
        # We only use Node Type as Features
        node_type = node_feat[0]
        type_index, nlen = self.tokenize_sentence(node_type, node_vd, eos=False)
        return type_index, nlen

    def get_textual_feat(self, node_feat, token_vd):
        """
        Sample Node Feature: ['Function', '', 0, False]
        :param node_feat: The four elements of Node Item
        :param token_vd: Token Vocab Dictionary
        :return:
        """
        # Use the Code String as the feature
        code_str = node_feat[1]
        tk_code_str = self.t3_parser.tokenize(code_str)
        nidxes, nlen = self.tokenize_sentence(tk_code_str, token_vd, eos=False)

        if not nidxes:
            # If there is no code in the node, we use an empty code placeholder
            nidxes.append(token_vd.get_w2i("<EMPTY_CODE>"))
            nlen = 1
        nidxes = pad_to_max(nidxes, self.config.max_code_token_len, pad_token=None)
        if nlen > self.config.max_code_token_len:
            nlen = self.config.max_code_token_len
        return nidxes, nlen



