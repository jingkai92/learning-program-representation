import os
import torch

from tokenizer.vocab_dict import TokenizationConstant
from utils.util import print_msg
from utils.pymodels_util import to_cuda
from factory.model_factory import ModelFactory
from evaluation.evaluator.resultsaver import ResultSaver


class BaseTrainer:
    def __init__(self, config):
        """
        Base Class for Trainer. Inherit this class for
        single_vocab_sequence method
        """
        self.config = config
        self.name = "BaseTrainer"
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.evaluator = None
        self.resultsaver = ResultSaver(config)
        self.patience = self.config.patience
        self.monitor_var = self.config.mm_var

    def setup_model(self):
        """
        Get model using self.experiment_type and ModelFactory
        :return:
        """
        model_class = ModelFactory().get_model(self.config)
        assert model_class is not None, "Model Factory fails to get Model Class"
        self.config.logger.info("Model: %s" % model_class)
        self.model = to_cuda(model_class(self.config), self.config.use_cuda)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.config.logger.info("Parameter Count: %i" % pytorch_total_params)
        self.update_model_params(pytorch_total_params)

        # Default Loss Function
        self.loss_function = self._get_loss_fn()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

    def _get_optimizer(self):
        # Optimizer might have different default weight decay
        if hasattr(self.config, 'weight_decay'):
            return self.get_wd_optim()
        else:
            return self.get_optim()

    def get_wd_optim(self):
        """
        Get the Optimizer with input weight decay
        :return:
        """
        if self.config.optimizer_type == "SGD":
            self.config.logger.info("Optimizer: SGD")
            return torch.optim.SGD(self.model.parameters(), lr=self.config.lr,
                                   weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type == "ADAMW":
            self.config.logger.info("Optimizer: ADAMW")
            return torch.optim.AdamW(self.model.parameters(), lr=self.config.lr,
                                     weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type == "ADAMAX":
            self.config.logger.info("Optimizer: ADAMAX")
            return torch.optim.Adamax(self.model.parameters(), lr=self.config.lr,
                                      weight_decay=self.config.weight_decay)
        else:
            self.config.logger.info("Optimizer: ADAM")
            return torch.optim.Adam(self.model.parameters(), lr=self.config.lr,
                                    weight_decay=self.config.weight_decay)

    def get_optim(self):
        """
        Get the Optimizer without WD
        :return:
        """
        if self.config.optimizer_type == "SGD":
            # Default Weight Decay is 0
            self.config.logger.info("Optimizer: SGD")
            return torch.optim.SGD(self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer_type == "ADAMW":
            # Default Weight Decay is 1e-2
            self.config.logger.info("Optimizer: ADAMW")
            return torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer_type == "ADAMAX":
            self.config.logger.info("Optimizer: ADAMAX")
            # Default Weight Decay is 1e-8
            return torch.optim.Adamax(self.model.parameters(), lr=self.config.lr)
        else:
            # Default Weight Decay is 0
            self.config.logger.info("Optimizer: ADAM")
            return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def _get_scheduler(self):
        if self.config.use_scheduler:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                              factor=0.5, patience=5, verbose=False)
        else:
            return None

    def _get_loss_fn(self):
        """
        Trainer Subclass should implement this
        :return:
        """
        raise Exception("SubClass should implement this.")

    def _initialize_best_scores(self, key_list):
        """
        Initialize the dictionary that will store the best scores
        :return:
        """
        if self.best_scores is None:
            self.best_scores = dict()
            for key in key_list:
                self.best_scores[key] = -1

    def _update_best_scores(self, epoch_num, val_scores):
        """
        We update the score_dict if it is better than our last epoch
        :param epoch_num: Current Epoch Number
        :param val_scores: Validation Result
        :return: Return True if update happens or else return False
        """
        if val_scores[self.monitor_var] > self.best_scores[self.monitor_var]:
            old_best = self.best_scores[self.monitor_var]
            new_best = val_scores[self.monitor_var]
            self.patience = self.config.patience
            self.best_scores = val_scores
            self.update_best_epoch(epoch_num)
            self.saved_path = self.save_pymodel(self.model)
            self.config.logger.info("{} improved:{:.4f}->{:.4f} ".format(self.monitor_var, old_best, new_best))
            self.update_val_results(val_scores)
            return True
        else:
            self.patience = self.patience - 1
            return False

    # def print_msg(self, msg):
    #     """
    #     Print the msg in a formatted way
    #     :param msg: Msg to be printed
    #     :return:
    #     """
    #     print_msg(msg, name=self.name)

    def save_pymodel(self, model):
        """
        Save model to <self.config.output_model>/self.name/timestamp.pt
        :return: Return the path of the saved model
        """
        if self.config.save_output:
            model_path = self.config.output_path
            path = os.path.join(model_path, "model.pt")
            torch.save(model.state_dict(), path)
            self.config.logger.info("Saving the best model to %s" % path)
            return path
        return None

    def load_pymodel(self, model_path):
        """
        Save model to <self.config.output_model>/self.name/timestamp.pt
        :return: Return the path of the saved model
        """
        if not self.model:
            return None
        self.print_msg("Loading Model from %s" % model_path)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        return self.model

    def _convert_seq_to_word(self, seqs, tokenizer, to_string=False):
        """
        Convert the single_vocab_sequence (In Integer) back to String of Words
        :param seqs: List of List of Integer
        :param tokenizer: Tokenizer
        :return: Return a list of list of string
        """
        translated_seq = self._convert_to_word(seqs, tokenizer)
        if to_string:
            return [" ".join(x) for x in translated_seq]
        return translated_seq

    def _convert_seqs_to_word(self, seqs, tokenizer, to_string=False):
        """
        Convert the single_vocab_sequence (In Integer) back to String of Words
        :param seqs: List of List of Integer
        :param tokenizer: Tokenizer
        :return: Return a list of list of string
        """
        translated_seqs = []
        for item in seqs:
            translated_seqs.append(self._convert_seq_to_word(item, tokenizer, to_string))
        return translated_seqs

    def _convert_to_word(self, seqs, tokenizer):
        """
        Convert to word using our own tokenizer, NON-BPE
        :param tokenizer: Tokenizer Object
        :return: Return sentence in string
        """
        if tokenizer.is_bpe():
            seqs = self._filter_non_word(seqs)
            dec_seq = tokenizer.convert_ids_to_sentence(seqs)
            return [x.split() for x in dec_seq]

        str_list = []
        for item in seqs:
            seq_string = []
            for num in item:
                if num == tokenizer.get_w2i("PAD"):
                    continue
                if num == tokenizer.get_w2i("SOS"):
                    continue
                if num == tokenizer.get_w2i("EOS"):
                    break
                seq_string.append(tokenizer.get_i2w(num))
            str_list.append(seq_string)
        return str_list

    def update_train_results(self, train_scores):
        """
        Update the latest train result into ResultSaver
        :param train_scores: Training Metrics
        :return: NIL
        """
        if self.resultsaver is None:
            return
        # Always the latest
        self.resultsaver.train_metrics = train_scores

    def update_val_results(self, val_scores):
        """
        Only get the best validation score
        :param val_scores: Validation Metrics
        :return: NIL
        """
        if self.resultsaver is None:
            return
        # Only the best validation score
        self.resultsaver.val_metrics = val_scores

    def update_test_results(self, test_scores):
        """
        Update the test result, only called one
        :param test_scores: Test Score Metrics
        :return: NIL
        """
        if self.resultsaver is None:
            return
        # Only the best validation score
        self.resultsaver.test_metrics = test_scores

    def update_time_metric(self, time_elapsed):
        """
        Update the metrics on time
        :param time_elapsed: Total Time elapsed
        :return:
        """
        if self.resultsaver is None:
            return
        # Only the best validation score
        self.resultsaver.time_elapsed = time_elapsed

    def update_best_epoch(self, epoch_num):
        """
        Update the resultsaver max epoch
        :param epoch_num:
        :return:
        """
        if self.resultsaver is None:
            return
        # Only the best validation score
        self.resultsaver.max_epoch = epoch_num

    def update_model_params(self, params_count):
        """
        Update the resultsaver max epoch
        :param params_count:
        :return:
        """
        if self.resultsaver is None:
            return
        # Only the best validation score
        self.resultsaver.model_params = params_count

    def pretty_print_score(self, scores, running_mode):
        """
        Pretty Print all the scores
        :param scores: Score of the training result
        :param running_mode: train or test or val
        :return:
        """
        assert self.evaluator is not None
        metric_str = self.evaluator.get_pretty_metric(scores)
        rm_str = "[{:^5}] ".format(running_mode)
        self.config.logger.info(rm_str + metric_str)

    @staticmethod
    def _filter_non_word(seqs):
        new_seqs = []
        for seq in seqs:
            new_seq = []
            for word_id in seq:
                if word_id == int(TokenizationConstant.PAD.value):
                    continue
                if word_id == int(TokenizationConstant.EOS.value):
                    continue
                if word_id == int(TokenizationConstant.SOS.value):
                    continue
                new_seq.append(word_id)
            new_seqs.append(new_seq)
        return new_seqs
