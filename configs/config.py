import os
import yaml
from configs.model_type import ModelType
from configs.task_type import TaskType, Task
from utils.util import print_msg, check_if_path_exists, load_vocab_dict, get_logger, timestamp, torch_setup


class Config:
    def __init__(self, yaml_config_path, test_mode=False):
        """
        Convert a YAML Config File into a Config object
        :param yaml_config_path: Path of YAML Config
        :param test_mode: Specify if you are test
        """
        self.name = "NIL"
        self.class_weights = None
        # Essential Configuration that cannot be missed.
        self.output_path = ""
        self.save_output = True
        self.test_mode = False
        check_if_path_exists(yaml_config_path)
        with open(yaml_config_path) as f:
            # use safe_load instead load
            config = yaml.safe_load(f)
        self.__dict__.update(**config)

        # Here marks the end of converting of yml attribute to config class attributes
        self.token_vocab_dict = None
        self.node_vocab_dict = None
        self.target_vocab_dict = None
        self.word_vocab_dict = None
        if not self.output_path:
            self.logger.warn("Output Path is not specified. Defaults to ./output")
            self.output_path = "./output"

        self.timestamp = timestamp
        log_dir_path = "./log"
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        self.log_path = os.path.join("./log", self.timestamp + ".log")
        self.logger = self.get_logger()
        self.torch_setup()
        if not test_mode:
            self.output_path = os.path.join(self.output_path, "%s" % self.timestamp)
        if self.save_output is not False:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

        # Project/Experiment Specified Configuration
        self.model_type = ModelType[self.model_type]
        self.task_type = TaskType[self.task_type]
        self.task = Task[self.task]
        self.creport = False
        self.evaluate = False
        # Save a copy of the YML file into output path
        if self.save_output:
            with open(os.path.join(self.output_path, 'args.yml'), 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)

    def print_params(self):
        """
        Print out all the attribute in the config object
        :return:
        """
        self.logger.info("=========================== Configuration ===========================")
        attr_dict = vars(self)
        for key, item in attr_dict.items():
            self.logger.info("\t%s: %s" % (key, item))
        self.logger.info("=" * 100)

    def torch_setup(self):
        """
        Call the torch setup in util.py
        :return:
        """
        self.logger.info("GPU ID: %s" % self.gpu_id)
        self.logger.info("Seed: %i" % 4096)
        self.logger.info("Pytorch Threads: %i" % 8)
        torch_setup(self.gpu_id, seed=4096, num_threads=8)

    def get_logger(self):
        """
        Get the logger object for better logging
        :return: NIL
        """
        return get_logger(self.log_path)

    def set_token_vocab_dict(self, vocab_dict):
        """
        Set the Vocab Dictionary, Ease the retrieving of vocab dictionary later in the model
        :param vocab_dict: Vocabulary Dictionary
        :return:
        """
        self.token_vocab_dict = vocab_dict

    def set_node_vocab_dict(self, vocab_dict):
        """
        Set the Vocab Dictionary, Ease the retrieving of vocab dictionary later in the model
        :param vocab_dict: Vocabulary Dictionary
        :return:
        """
        self.node_vocab_dict = vocab_dict

    def set_target_vocab_dict(self, vocab_dict):
        """
        Set the Vocab Dictionary, Ease the retrieving of vocab dictionary later in the model
        :param vocab_dict: Vocabulary Dictionary
        :return:
        """
        self.target_vocab_dict = vocab_dict

    def set_word_vocab_dict(self, vocab_dict):
        """
        Set the Vocab Dictionary, Ease the retrieving of vocab dictionary later in the model
        :param vocab_dict: Vocabulary Dictionary
        :return:
        """
        self.word_vocab_dict = vocab_dict

    def print_msg(self, msg):
        """
        Print Message by using Util print_msg method
        :param msg: Message to be printed out
        :return:
        """
        print_msg(msg, name=self.name)

    # Project Specified Method
    def set_class_weights(self, class_weights):
        """
        Set the class weights for a single data
        :param class_weights:
        :return:
        """
        self.class_weights = class_weights

    @property
    def edge_type_list(self):
        """
        Get the edge type based on self.config.ggnn['use_edge_type']
        Set the list of edge type, in number.
        :return:
        """
        if self.use_edge_type == "cpg":
            return [0, 1, 2, 3, 4, 5, 6, 7]
        elif self.use_edge_type == "ast":
            return [0]
        elif self.use_edge_type == "cfg":
            return [3]
        elif self.use_edge_type == "cdg":
            return [7]
        elif self.use_edge_type == "ddg":
            return [4, 5, 6]
        else:
            raise NotImplementedError

    def setup_vocab_dict(self):
        """
        Setup vocab dictionary for all the data
        :return:
        """

        if hasattr(self, "token_vocab_path"):
            token_vocab_dict = load_vocab_dict(self.token_vocab_path)
            self.set_token_vocab_dict(token_vocab_dict)
            self.logger.info("Loading dictionary: %s/%i" % (self.token_vocab_path, len(token_vocab_dict)))

        if hasattr(self, "node_vocab_path"):
            node_vocab_dict = load_vocab_dict(self.node_vocab_path)
            self.set_node_vocab_dict(node_vocab_dict)
            self.logger.info("Loading dictionary: %s/%i" % (self.node_vocab_path, len(node_vocab_dict)))

        if hasattr(self, "target_vocab_path"):
            target_vocab_dict = load_vocab_dict(self.target_vocab_path)
            self.set_target_vocab_dict(target_vocab_dict)
            self.logger.info("Loading dictionary: %s/%i" % (self.target_vocab_path, len(target_vocab_dict)))

        if hasattr(self, "word_vocab_path"):
            word_vocab_dict = load_vocab_dict(self.word_vocab_path)
            self.set_word_vocab_dict(word_vocab_dict)
            self.logger.info("Loading dictionary: %s/%i" % (self.word_vocab_path, len(word_vocab_dict)))

    def get_vocab_size_for_embedding(self):
        """
        If it is Code Classification and Vulnerability Detection,
        we return token_vocab_dict.vocab_size()
        If it is patch identification, we return word_vocab_dict.vocab_size()
        :return:
        """
        if self.task in [Task.CodeClassification, Task.VulnerabilityDetection]:
            return self.token_vocab_dict.vocab_size()
        elif self.task in [Task.PatchIdentification]:
            return self.word_vocab_dict.vocab_size()
        raise Exception("Embedding Vocab Size not found.")
