import dgl
import torch

from bases.base_formatter import BaseFormatter
from bases.base_graph_formatter import BaseGraphFormatter
from tokenizer.code_tokenizer import CodeTokenizer
from tokenizer.vocab_dict import TokenizationConstant
from utils.pymodels_util import pad_to_max, stack_seq_to_tensor, to_cuda


class SingleEdgeGraphPairwiseFormatter(BaseGraphFormatter):
    def __init__(self, config, name="SingleEdgeGraphPairwiseFormatter"):
        """
        SingleEdgeGraphPairwiseFormatter will format the input data.
        """
        self.name = name
        self.disable_tqdm = config.disable_tqdm
        self.config = config
        BaseGraphFormatter.__init__(self, config, name)

    def format(self, item_json, vocab_dicts):
        """
        Format single item_json using the Vocab Dictionary
        :param item_json: JSON of a single item in the dataset
        :param vocab_dicts: ["Token", "Node", "Target"].
        :return: Return Datapoints
        """
        token_vd, node_vd, target_vd, word_vd = vocab_dicts
        datapoint = self.datapoint_class()
        # Target
        datapoint.tgt = item_json['target']
        # First Graph
        dgl_graph_one = self.get_graph(item_json['item_1'], token_vd, node_vd)
        datapoint.function_one_graph = dgl_graph_one
        # Second Graph
        dgl_graph_two = self.get_graph(item_json['item_2'], token_vd, node_vd)
        datapoint.function_two_graph = dgl_graph_two
        return datapoint

    def get_graph(self, cur_item, token_vd, node_vd):
        """
        Get the graph of the current item
        :param cur_item: One item
        :param token_vd: Token Vocab Dictionary
        :param node_vd: Node Vocab Dictionary
        :return:
        """
        graph = self._convert_to_single_edge_graph(cur_item['jsgraph'], token_vd, node_vd)
        return graph
