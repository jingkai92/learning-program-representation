import os
import time
import statistics
from tqdm import tqdm
from torch.utils.data import DataLoader
from factory.collate_factory import CollateFactory
from factory.formatter_factory import FormatterFactory
from utils.util import load_json, load_gzip_as_json, is_tfidf


class BaseDataset:
    def __init__(self, name, config, running_mode):
        """
        Base Class for Dataset
        """
        self.name = name
        self.config = config
        self.running_mode = running_mode
        self.dataformatter = FormatterFactory().get_formatter(self.config)
        self.config.logger.info("DataFormatter: %s" % self.dataformatter.name)

        # Loading the train/val/test data
        # Default loading method is GZIP - Save space
        train_path = os.path.join(config.dataset['path'], 'train.gzip')
        val_path = os.path.join(config.dataset['path'], 'val.gzip')
        test_path = os.path.join(config.dataset['path'], 'test.gzip')
        if running_mode == "train":
            self.train_jsons = self._load_data_from_json(train_path, gzip_format=True)
            self.val_jsons = self._load_data_from_json(val_path, gzip_format=True)
        if running_mode == "test" or self.is_tfidf():
            self.test_jsons = self._load_data_from_json(test_path, gzip_format=True)
        self.train_datapoints, self.val_datapoints, self.test_datapoints = [], [], []
        self.format_data()
        if self.config.class_weight and self.running_mode == "train":
            self.set_class_weights()

    def _load_data_from_json(self, js_path, gzip_format=True):
        """
        Load the data from the JSON Path
        :param js_path: Path
        :param gzip_format: Specify if it is a gzip
        :return:
        """
        start = time.time()
        if gzip_format:
            json_object = load_gzip_as_json(js_path)
        else:
            json_object = load_json(js_path)
        end = time.time()
        total_time = end - start
        self.config.logger.info("Path: %s, Size: %i, Time: %.2f" % (js_path, len(json_object), total_time))
        return json_object

    def format_data(self):
        """
        Default Format for data. If you need any changes, you can
        use the sub-class to overload this function
        :return: NIL
        """
        # Use a single batch to test if our data is correctly format - For testing only
        if self.config.initial_test:
            # Test a single batch
            if self.running_mode == "train":
                self.train_jsons = self.train_jsons[:self.config.batch_size]
                self.val_jsons = self.val_jsons[:self.config.batch_size]
            if self.running_mode == "test" or self.is_tfidf():
                self.test_jsons = self.test_jsons[:self.config.batch_size]

        if self.running_mode == "train":
            self.train_datapoints = self._format(self.train_jsons)
            self.val_datapoints = self._format(self.val_jsons)
        if self.running_mode == "test" or self.is_tfidf():
            self.test_datapoints = self._format(self.test_jsons)

    def set_class_weights(self):
        """
        Get the class weight
        :return: NIL
        """
        class_weights = self.get_class_weights()
        self.config.set_class_weights(class_weights)
        self.config.logger.info("Total Missing Word: %i" % self.dataformatter.missing_word_count)

    def get_class_weights(self):
        """
        Default getting class weight method
        :return: Return the dictionary for class weights
        """
        count_dict = dict()
        for dp in self.train_datapoints:
            tgt = dp.tgt
            if tgt not in count_dict:
                count_dict[tgt] = 0
            count_dict[tgt] += 1
        all_values = count_dict.values()
        min_value = min(all_values)
        class_weights = [0] * len(count_dict)
        for key, value in count_dict.items():
            class_weights[key] = float(min_value) / value
        return class_weights

    def _format(self, jsons):
        """
        Format the JSONS into their respective datapoint
        using data formatter
        :param jsons: List of JSONs
        :return:
        """
        datapoints = []
        for item in tqdm(jsons, disable=self.config.disable_tqdm, desc="Formatting Data"):
            datapoint = self.dataformatter.format(item, self.get_vocabs())
            datapoints.append(datapoint)
        return datapoints

    def get_dls(self):
        """
        Get the train data loader and validate data loader according to experiment type.
        :return: Return train_dl and val_dl
        """
        self.config.logger.info("Train Datapoints: %i" % len(self.train_datapoints))
        self.config.logger.info("Val Datapoints: %i" % len(self.val_datapoints))
        collate_fn = CollateFactory().get_collate_fn(self.config)
        assert self.train_datapoints is not None
        assert self.val_datapoints is not None
        train_dl = self._set_datapoints_dataloader(self.train_datapoints,
                                                   collate_fn, shuffle=True)
        val_dl = self._set_datapoints_dataloader(self.val_datapoints,
                                                 collate_fn, shuffle=False)
        return train_dl, val_dl

    def get_testing_dl(self):
        """
        Get the testing data loader
        :return: Return data loader
        """
        collate_fn = CollateFactory().get_collate_fn(self.config)
        self.config.logger.info("Test Datapoints: %i" % len(self.test_datapoints))
        assert self.test_datapoints
        test_dl = self._set_datapoints_dataloader(self.test_datapoints, collate_fn, shuffle=False)
        return test_dl

    def get_train_dl(self, shuffle=True):
        """
        Get the training data loader
        :return: Return data loader
        """
        collate_fn = CollateFactory().get_collate_fn(self.config)
        assert self.train_datapoints
        train_dl = self._set_datapoints_dataloader(self.train_datapoints,
                                                   collate_fn, shuffle=shuffle)
        return train_dl

    def is_tfidf(self):
        """
        Check if the current experiment mode is TFIDF mode
        :return:
        """
        return is_tfidf(self.config.model_type)

    def _set_datapoints_dataloader(self, datapoints, collate_fn, shuffle=False):
        """
        Initiate the Data loader object
        :param datapoints: Source Data
        :param collate_fn: Collate Function for retrieving batches
        :param shuffle: Specify if you want to shuffle the data
        :return: Return the Dataloader
        """
        tmp_dataset = DatapointDataset(datapoints)
        tmp_dl = DataLoader(tmp_dataset, batch_size=self.config.batch_size,
                            shuffle=shuffle, collate_fn=collate_fn, num_workers=0)
        return tmp_dl

    def get_statistic(self, datapoints):
        """
        Get statistic for Dataset
        :param datapoints: Datapoint List
        :return: NIL
        """
        lens = []
        for dp in datapoints:
            lens.append(len(dp.func_lines))
        avg_line_num = sum(lens) / float(len(lens))
        self.print_msg("Printing Statistic")
        self.print_msg("Average Length of the Function: %f" % avg_line_num)
        self.print_msg("Median Length of the Function: %f" % statistics.median(lens))
        self.print_msg("Max Length of the Function: %f" % max(lens))
        self.print_msg("Min Length of the Function: %f" % min(lens))
        lt5 = 0
        for item in lens:
            if item < 5:
                lt5 += 1
        self.print_msg("Number of Function less than 5 Lines: %i" % lt5)

    @staticmethod
    def _clean_buggy_function(buggy_function, bug_location):
        """
        Clean the buggy function such that there will be no empty lines
        :param buggy_function: Buggy Function where lines is in a list
        :param bug_location: Bug Location, i.e., line number of the bug
        :return: Return a buggy function that are cleaned
        """
        clean_lines = []
        total_offset = 0
        for i, line in enumerate(buggy_function):
            if not line.strip():
                assert i != bug_location
                if bug_location > i:
                    total_offset += 1
                continue
            clean_lines.append(line)
        return clean_lines, bug_location - total_offset

    def get_vocabs(self):
        """
        Return all three vocab dict in the following order
        ["Token", "Node", "Target"].
        :return:
        """
        return [self.config.token_vocab_dict,
                self.config.node_vocab_dict,
                self.config.target_vocab_dict,
                self.config.word_vocab_dict]


class DatapointDataset:
    def __init__(self, datapoints):
        """
        For loading using the PyTorch DataLoader
        :param datapoints: Data Points
        """
        self.datapoints = datapoints

    def __getitem__(self, idx):
        batch_item = self.datapoints[idx]
        return batch_item

    def __len__(self):
        return len(self.datapoints)
