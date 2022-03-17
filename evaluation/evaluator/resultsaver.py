from tabulate import tabulate


class ResultSaver:
    def __init__(self, config):
        """
        Format the result out nicely so I can paste it into Google Docs better lol
        """
        self.config = config
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None
        self.time_elapsed = None
        self.max_epoch = None
        self.model_params = None

    def pretty_print_score(self):
        """
        Pretty Print all the three scores
        :return:
        """
        if self.config.classify_mode == "Binary":
            self.print_binary_mode()
        elif self.config.classify_mode == "Summarization":
            self.print_summarize_mode()
        elif self.config.classify_mode == "Multi-class":
            self.print_multiclass_mode()
        else:
            raise Exception

    def print_binary_mode(self):
        """
        We will follow a certain hardcoded format
        TRAIN:
            Training Loss	Training Accuracy	Training Precision	Training Recall	Training F1
        VAL:
            Val Loss	Val Accuracy	Val Precision	Val Recall	Val F1
        TEST:
            Testing Accuracy	Testing Precision	Testing Recall	Testing F1
        :return:
        """
        format_cols = ['', 'loss', 'accuracy', 'precision', 'recall', 'f1']
        train_row = self.format_dict_to_str(self.train_metrics, format_cols, name='train')
        val_row = self.format_dict_to_str(self.val_metrics, format_cols, name='val')
        test_row = self.format_dict_to_str(self.test_metrics, format_cols, name='test')
        print("-" * 100)
        print(tabulate([train_row, val_row, test_row], headers=format_cols, tablefmt="presto"))
        print("-" * 100)
        print(tabulate([[self.time_elapsed, self.max_epoch, self.model_params]], headers=['Time', "Max Epoch",
                                                                                          "Model Params"],
                       tablefmt="presto"))

    def print_multiclass_mode(self):
        """
        We will follow a certain hardcoded format
        TRAIN:
            Training Loss	Training Accuracy	Training Precision	Training Recall	Training F1
        VAL:
            Val Loss	Val Accuracy	Val Precision	Val Recall	Val F1
        TEST:
            Testing Accuracy	Testing Precision	Testing Recall	Testing F1
        :return:
        """
        format_cols = ['', 'loss', 'weighted_accuracy']
        train_row = self.format_dict_to_str(self.train_metrics, format_cols, name='train')
        val_row = self.format_dict_to_str(self.val_metrics, format_cols, name='val')
        test_row = self.format_dict_to_str(self.test_metrics, format_cols, name='test')
        print("-" * 100)
        print(tabulate([train_row, val_row, test_row], headers=format_cols, tablefmt="presto"))
        print("-" * 100)
        print(tabulate([[self.time_elapsed, self.max_epoch, self.model_params]], headers=['Time', "Max Epoch",
                                                                                          "Model Params"],
                       tablefmt="presto"))

    def print_summarize_mode(self):
        """
        We will follow a certain hardcoded format
        TRAIN:
            Training Loss	RAcc
        VAL:
            Val Loss	RAcc
        TEST:
            Testing Accuracy	RAcc
        :return:
        """
        format_cols = ['', 'loss', 'Precision', 'Recall', 'F1 Score']
        train_row = self.format_dict_to_str(self.train_metrics, format_cols, name='train')
        val_row = self.format_dict_to_str(self.val_metrics, format_cols, name='val')
        test_row = self.format_dict_to_str(self.test_metrics, format_cols, name='test')
        print("-" * 100)
        print(tabulate([train_row, val_row, test_row], headers=format_cols, tablefmt="presto"))
        print("-" * 100)
        print(tabulate([[self.time_elapsed, self.max_epoch, self.model_params]], headers=['Time', "Max Epoch",
                                                                                          "Model Params"],
                       tablefmt="presto"))

    @staticmethod
    def format_dict_to_str(metrics, format_cols, name="train"):
        """
        Format the metrics dictionary to the format cols
        :param metrics: Metric Dictionary
        :param format_cols: Format Columns
        :param name: Either is train, val or test
        :return: Return a single list of cols
        """
        row = [name]
        for item in format_cols[1:]:
            key = item
            if item == "loss":
                key = "%s loss" % name
            value = metrics[key]
            out_str = "{:.4f}".format(value)
            row.append(out_str)
        assert len(row) == len(format_cols)
        return row
