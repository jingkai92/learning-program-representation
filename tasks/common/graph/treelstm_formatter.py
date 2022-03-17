import dgl
import torch

from bases.base_formatter import BaseFormatter
from bases.base_graph_formatter import BaseGraphFormatter
from tokenizer.code_tokenizer import CodeTokenizer
from tokenizer.vocab_dict import TokenizationConstant
from utils.pymodels_util import pad_to_max, stack_seq_to_tensor, to_cuda


class TreeLSTMFormatter(BaseGraphFormatter):
    def __init__(self, config, name="TreeLSTMFormatter"):
        """
        TreeLSTMFormatter will format the input data.
        """
        self.name = name
        self.disable_tqdm = config.disable_tqdm
        self.config = config
        self.t3_parser = CodeTokenizer(data=[], lang="C", tlevel='t3')
        BaseFormatter.__init__(self, config, name)

    def format(self, item_json, vocab_dicts):
        """
        Format single item_json using the Vocab Dictionary
        {'IS_AST_PARENT': 0, 'isCFGParent': 1, "POST_DOM": 2,
        "FLOWS_TO": 3, "USE": 4, "DEF": 5, 'REACHES': 6, "CONTROLS": 7}
        :param item_json: JSON of a single item in the dataset
        :param vocab_dicts: ["Token", "Node", "Target"].
        :return: Return Datapoints
        """
        token_vd, node_vd, target_vd, word_vd = vocab_dicts
        datapoint = self.datapoint_class()
        datapoint.function = item_json['function']
        dgl_graph = self._convert_to_multi_edge_dglgraph(item_json['jsgraph'], token_vd, node_vd)
        datapoint.function_graph = dgl_graph
        datapoint.graph_size = item_json['graph_size']
        if type(item_json['target']) == int:
            datapoint.tgt = item_json['target']
        else:
            tok_tgt = self.t3_parser.tokenize(item_json['target'])
            datapoint.tgt = tok_tgt
            tok_tgt, blen = self.tokenize_sentence(tok_tgt, token_vd)
            datapoint.tgt_vec = tok_tgt
        return datapoint
