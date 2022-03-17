import dgl
import torch

from bases.base_formatter import BaseFormatter
from bases.base_graph_formatter import BaseGraphFormatter
from tokenizer.code_tokenizer import CodeTokenizer
from tokenizer.vocab_dict import TokenizationConstant
from utils.pymodels_util import pad_to_max, stack_seq_to_tensor, to_cuda


class TreeLSTMPairwiseFormatter(BaseGraphFormatter):
    def __init__(self, config, name="TreeLSTMPairwiseFormatter"):
        """
        TreeLSTMFormatter will format the input data.
        """
        self.name = name
        self.disable_tqdm = config.disable_tqdm
        self.config = config
        self.t3_parser = CodeTokenizer(data=[], lang="C", tlevel='t3')
        BaseGraphFormatter.__init__(self, config, name)

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
        # Target
        datapoint.tgt = item_json['target']

        dgl_graph_one = self.get_graph(item_json['item_1'], token_vd, node_vd)
        datapoint.function_one_graph = dgl_graph_one

        dgl_graph_two = self.get_graph(item_json['item_2'], token_vd, node_vd)
        datapoint.function_two_graph = dgl_graph_two
        # datapoint.graph_size = item_json['graph_size']
        return datapoint

    def get_graph(self, cur_item, token_vd, node_vd):
        """
        Get the graph of the current item
        :param cur_item: One item
        :param token_vd: Token Vocab Dictionary
        :param node_vd: Node Vocab Dictionary
        :return:
        """
        graph = self._convert_to_multi_edge_dglgraph(cur_item['jsgraph'], token_vd, node_vd)
        return graph

