from bases.base_formatter import BaseFormatter


class TFIDFPairwiseFormatter(BaseFormatter):
    def __init__(self, config, name="TFIDFPairwiseFormatter"):
        """
        Formatter for TFIDF on Pairwise Classification
        :param config: Configuration Object
        :param name: Name of the Formatter
        """
        BaseFormatter.__init__(self, config, name)

    def format(self, item_json, vocab_dicts):
        """
        Format single item_json using the Vocab Dictionary
        :param item_json: JSON of a single item in the dataset
        :param vocab_dicts: ["Token", "Node", "Target"].
        :return: Return Datapoints
        """
        datapoint = self.datapoint_class()
        item_1 = item_json['item_1']
        item_2 = item_json['item_2']
        item_fn_1 = self.format_function(item_1)
        item_fn_2 = self.format_function(item_2)
        datapoint.tgt = item_json['target']
        datapoint.function_one = item_fn_1
        datapoint.function_two = item_fn_2
        return datapoint

    def format_function(self, clone_json):
        """
        Format the function and return a string of tokenized and processed item
        :param clone_json: Clone JSON
        :return:
        """
        fn = self.t3_parser.tokenize(clone_json['function'])
        return " ".join(fn.split()[:self.config.max_function_length])
