from bases.base_formatter import BaseFormatter
from tokenizer.code_tokenizer import CodeTokenizer


class TFIDFFormatter(BaseFormatter):
    def __init__(self, config, name="TFIDFFormatter"):
        """
        TFIDFFormatter will format the input data
        """
        self.name = name
        self.disable_tqdm = config.disable_tqdm
        self.config = config
        self.t3_parser = CodeTokenizer(data=[], lang="C", tlevel='t3')
        BaseFormatter.__init__(self, config, name)

    def format(self, item_json, vocab_dicts):
        """
        Format single item_json using the Vocab Dictionary
        :param item_json: JSON of a single item in the dataset
        :param vocab_dicts: ["Token", "Node", "Target"].
        :return: Return Datapoints
        """
        datapoint = self.datapoint_class()
        datapoint.function = self.t3_parser.tokenize(item_json['function'][:self.config.max_function_length])
        datapoint.tgt = item_json['target']
        return datapoint
