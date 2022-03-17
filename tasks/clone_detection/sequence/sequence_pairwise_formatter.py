from bases.base_formatter import BaseFormatter


class SequencePairwiseFormatter(BaseFormatter):
    def __init__(self, config, name="SequencePairwiseFormatter"):
        """
        Formatter for Sequence on Pairwise Classification
        :param config: Configuration Object
        :param name: Name of the Formatter
        """
        BaseFormatter.__init__(self, config, name)

    def format(self, item_json, vocab_dicts):
        """
        Format single item_json using the Vocab Dictionary
        :param item_json: JSON of a single item in the dataset
        :param vocab_dicts:  ["Token", "Node", "Target"].
        :return: Return Datapoints
        """
        token_vd, node_vd, target_vd, word_vd = vocab_dicts
        datapoint = self.datapoint_class()

        fn_one = self.format_function(item_json['item_1'])
        tok_function_one, blen_one = self.tokenize_sentence(fn_one, token_vd)
        datapoint.function_one_vec = tok_function_one

        fn_two = self.format_function(item_json['item_2'])
        tok_function_two, blen_one = self.tokenize_sentence(fn_two, token_vd)
        datapoint.function_two_vec = tok_function_two
        datapoint.tgt = item_json['target']
        return datapoint

    def format_function(self, clone_json):
        """
        Format the function and return a string of tokenized and processed item
        :param clone_json: Clone JSON
        :return:
        """
        fn = self.t3_parser.tokenize(clone_json['function'])
        return " ".join(fn.split()[:self.config.max_function_length])
