import time
import torch
import datetime
from tqdm import tqdm
import torch.nn as nn
from bases.base_trainer import BaseTrainer
from evaluation.evaluator.summarization_evaluator import SummarizationEvaluator
from tokenizer.vocab_dict import TokenizationConstant
from utils.pymodels_util import get_average
from utils.util import get_pretty_metric
from evaluation.evaluator.classification_evaluator import ClassificationEvaluator


class SummarizeTrainer(BaseTrainer):
    def __init__(self, config):
        BaseTrainer.__init__(self, config)
        self.name = "SummarizeTrainer"
        self.config = config
        self.timestamp = self.config.timestamp

        self.model = None
        self.best_scores = None
        self.saved_path = ""
        self.evaluator = SummarizationEvaluator(config)

    def _get_loss_fn(self):
        return nn.CrossEntropyLoss(ignore_index=int(TokenizationConstant.PAD.value))

    def start_train(self, dataset):
        """
        Start the training process. We put as base method so that we
        can do some printing for visualization
        :param dataset: Dataset Object for Training and Validating
        :return:
        """
        start = time.time()
        train_dl, val_dl = dataset.get_dls()
        self.print_msg("Start Training for %s" % self.name)
        for epoch_num in range(self.config.max_epoch):
            print()
            self.print_msg("Epoch Num: %s" % str(epoch_num))
            train_scores = self._run_dl(train_dl, running_mode="train")
            self.print_msg("[Training] %s" % get_pretty_metric(train_scores))
            self.update_train_results(train_scores)
            self.print_msg("=" * 100)
            with torch.no_grad():
                val_scores = self._run_dl(val_dl, running_mode="val")
            self.print_msg("[Validation] %s" % get_pretty_metric(val_scores))
            if self.scheduler:
                self.scheduler.step(val_scores['val loss'])
            self._initialize_best_scores(list(val_scores.keys()))
            updated = self._update_best_scores(epoch_num, val_scores)
            if self.patience == 0:
                break
            print()
        end = time.time()
        total_time = end - start
        time_elapsed = str(datetime.timedelta(seconds=total_time)).split(".")[0]
        self.print_msg("Total Time Elapsed: %s" % time_elapsed)
        self.update_time_metric(time_elapsed)

    def start_test(self, dataset):
        """
        Start the testing of the model
        :return:
        """
        test_dl = dataset.get_testing_dl()
        with torch.no_grad():
            test_scores = self._run_dl(test_dl, running_mode="test")
        # self.print_msg("[Testing] %s" % get_pretty_metric(test_scores))
        return test_scores

    def _run_dl(self, dl, running_mode="train"):
        """
        Run a single DL with self.model
        :param dl: Dataloader Object, Can be train_dl, val_dl or Test_dl
        :param running_mode: Specify if it is training the data in order to perform inference or not
        :return:
        """
        losses = []
        # start_time = time.time()
        for iter, batch_dict in tqdm(enumerate(dl), disable=self.config.disable_tqdm):
            if running_mode == "train":
                self.model.train()
                self.optimizer.zero_grad()
            else:
                self.model.eval()

            model_input, model_output, loss = self.model(batch_dict, running_mode, loss_fn=self.loss_function)
            if running_mode == "train":
                loss.backward()
                # Optimizer Step
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
                self.optimizer.step()

            if running_mode in ["train", 'val']:
                if type(loss) == int:
                    losses.append(loss)
                else:
                    losses.append(loss.item())
            ground_truth = [" ".join(item.split("_")) for item in batch_dict['tgt']]
            model_output = self.convert_model_output_to_string(model_output)
            src_string = self._convert_seq_to_word(model_input, self.config.token_vocab_dict, to_string=True)
            self.evaluator.add_strings(src_string, model_output, ground_truth)
        scores = self.evaluator.evaluate_score(name=running_mode)
        scores.update({"%s loss" % running_mode: get_average(losses)})
        # end_time = time.time()
        # time_needed = end_time - start_time
        # self.print_msg("Time Elapsed: %s" % str(datetime.timedelta(seconds=time_needed)).split(".")[0])
        return scores

    def convert_model_output_to_string(self, batch_model_output):
        """
        We use Target Vocab Dict if it exists or else we assume that it is
        unify dictionary
        :param batch_model_output: Batch Model Output from the Model
        :return: Return the translated output
        """
        if self.config.target_vocab_dict:
            model_output = self._convert_seq_to_word(batch_model_output, self.config.target_vocab_dict, to_string=True)
        else:
            model_output = self._convert_seq_to_word(batch_model_output, self.config.token_vocab_dict, to_string=True)
        return model_output
