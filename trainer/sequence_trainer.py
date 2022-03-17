import os
import time
import torch
import datetime
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from bases.base_trainer import BaseTrainer
from utils.pymodels_util import get_average
from utils.util import get_pretty_metric, save_json
from evaluation.evaluator.classification_evaluator import ClassificationEvaluator


class SequenceTrainer(BaseTrainer):
    def __init__(self, config):
        BaseTrainer.__init__(self, config)
        self.model = None
        self.best_scores = None
        self.saved_path = ""
        self.name = "SequenceTrainer"
        self.config = config
        self.timestamp = self.config.timestamp
        self.evaluator = ClassificationEvaluator(config)

    def _get_loss_fn(self):
        if self.config.class_weights:
            assert self.config.class_weights is not None
            self.config.logger.info("Class Weights: %s" % str(self.config.class_weights))
            class_weight = torch.tensor(self.config.class_weights, dtype=torch.float).cuda()
            return nn.CrossEntropyLoss(weight=class_weight)
        else:
            return nn.CrossEntropyLoss()

    def start_train(self, dataset):
        """
        Start the training process. We put as base method so that we
        can do some printing for visualization
        :param dataset: Dataset Object for Training and Validating
        :return:
        """
        start = time.time()
        train_dl, val_dl = dataset.get_dls()
        for epoch_num in range(self.config.max_epoch):
            self.config.logger.info("-" * 100)
            self.config.logger.info("Epoch Num: %s" % str(epoch_num))
            train_scores = self._run_dl(train_dl, running_mode="train")
            self.pretty_print_score(train_scores, running_mode="train")

            self.update_train_results(train_scores)

            with torch.no_grad():
                val_scores = self._run_dl(val_dl, running_mode="val")
            self.pretty_print_score(val_scores, running_mode="val")
            if self.scheduler:
                self.scheduler.step(val_scores['val loss'])
            self._initialize_best_scores(list(val_scores.keys()))
            updated = self._update_best_scores(epoch_num, val_scores)
            if self.patience == 0:
                break
            # self.config.logger.info("-" * 100)
        end = time.time()
        total_time = end - start
        time_elapsed = str(datetime.timedelta(seconds=total_time)).split(".")[0]
        self.config.logger.info("Total Time: %s" % time_elapsed)
        self.update_time_metric(time_elapsed)

    def start_test(self, dataset):
        """
        Start the testing of the model
        :return:
        """
        test_dl = dataset.get_testing_dl()
        # Get result only
        with torch.no_grad():
            test_scores = self._run_dl(test_dl, running_mode="test")
        return test_scores

    def start_evaluate(self, dataset):
        """
        Start the testing of the model
        :return:
        """
        test_dl = dataset.get_testing_dl()
        with torch.no_grad():
            test_scores = self._evaluate_dl(test_dl, running_mode="test")
        return test_scores

    def _run_dl(self, dl, running_mode="train"):
        """
        Run a single DL with self.model
        :param dl: Dataloader Object, Can be train_dl, val_dl or Test_dl
        :param running_mode: Specify if it is training the data in order to perform inference or not
        :return:
        """
        losses = []
        start_time = time.time()
        for iter_num, batch_dict in tqdm(enumerate(dl), disable=self.config.disable_tqdm):
            if running_mode == "train":
                self.model.train()
                self.optimizer.zero_grad()
            else:
                self.model.eval()
            probs, labels, loss = self.model(batch_dict, running_mode, loss_fn=self.loss_function)
            if running_mode == "train":
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # Optimizer Step
                self.optimizer.step()

            if running_mode in ["train", 'val']:
                if type(loss) == int:
                    losses.append(loss)
                else:
                    losses.append(loss.item())
            # values, indices = torch.max(probs, dim=-1)
            self.evaluator.add_metric_data(probs.cpu().tolist(), labels)
        scores = self.evaluator.evaluate_score()
        scores.update({"%s loss" % running_mode: get_average(losses)})
        end_time = time.time()
        time_needed = end_time - start_time
        # self.config.logger.info("Time Elapsed: %s" % str(datetime.timedelta(seconds=time_needed)).split(".")[0])
        return scores

    def _evaluate_dl(self, dl, running_mode="test"):
        """
        Run a single DL with self.model and check the function and output of the model
        Basically evaluate the result of the model
        :param dl: Dataloader Object, Can be train_dl, val_dl or Test_dl
        :param running_mode: Specify if it is training the data in order to perform inference or not
        :return:
        """
        assert running_mode == "test"
        result_dicts = []
        for iter_num, batch_dict in tqdm(enumerate(dl), disable=self.config.disable_tqdm):
            self.model.eval()
            probs, labels, loss = self.model(batch_dict, running_mode, loss_fn=self.loss_function)
            best_preds = np.asarray([np.argmax(line) for line in probs.cpu().tolist()])
            assert len(best_preds) == len(labels) == len(batch_dict['function'])
            for i in range(len(best_preds)):
                result_dict = dict()
                result_dict['pred'] = int(best_preds[i])
                result_dict['target'] = labels[i]
                result_dict['function'] = batch_dict['function'][i]
                if 'fid' in batch_dict:
                    result_dict['fid'] = batch_dict['fid'][i]
                result_dicts.append(result_dict)
            self.evaluator.add_metric_data(probs.cpu().tolist(), labels)
        scores = self.evaluator.evaluate_score(name=running_mode)
        scores.update({"%s loss" % running_mode: 0})

        self.save_classify_output(result_dicts)

        count = 0
        for item in result_dicts:
            pred = item['pred']
            target = item['target']
            fn = item['function']
            if pred != target: #len(fn.splitlines()) < 10:
                count += 1
        self.print_msg("Total: %i" % count)
        return scores

    def get_vectors(self, dataset, vector_type="train"):
        """
        Get the learnt representation of the TRAINED model
        :param vector_type: Type of subset you want
        :param dataset: Dataset Object for loading and handling data
        :return:
        """
        running_mode = "test"
        dl = dataset.get_testing_dl() if vector_type == "test" else dataset.get_train_dl(shuffle=False)
        self.print_msg("Getting %s Dataloader" % vector_type)

        learnt_reprs = []
        preds = []
        start_time = time.time()
        for iter_num, batch_dict in tqdm(enumerate(dl), disable=self.config.disable_tqdm):
            self.model.eval()
            probs, labels, loss = self.model(batch_dict, running_mode, loss_fn=self.loss_function)
            best_preds = np.asarray([np.argmax(line) for line in probs.cpu().tolist()])
            if hasattr(self.model, "cur_meanfeats"):
                learnt_reprs.extend(self.model.cur_meanfeats.cpu().tolist())
                preds.extend(best_preds)
            self.evaluator.add_metric_data(probs.cpu().tolist(), labels)
        scores = self.evaluator.evaluate_score(name=running_mode)
        scores.update({"%s loss" % running_mode: 0})
        end_time = time.time()
        time_needed = end_time - start_time
        self.print_msg("Time Elapsed: %s" % str(datetime.timedelta(seconds=time_needed)).split(".")[0])
        json_list = []
        assert len(learnt_reprs) == len(preds)
        for i in range(len(learnt_reprs)):
            tmp_dict = {'pred': int(preds[i]), 'reprs': learnt_reprs[i]}
            json_list.append(tmp_dict)
        reprs_path = os.path.join(self.config.output_path, "%s_%s_reprs.json" % (vector_type, self.model.name))
        save_json(json_list, reprs_path)
        self.print_msg("Representation saved to %s" % reprs_path)
        return scores

    def save_classify_output(self, results):
        output_path = os.path.join(self.config.output_path, "output.json")
        save_json(results, output_path)
        # output_path = os.path.join(self.config.output_path, "correct_output.txt")
        # count = 0
        # with open(output_path, 'w', errors='replace') as wfile:
        #     for item in results:
        #         if count > 200:
        #             break
        #         pred = item['pred']
        #         target = item['target']
        #         fn = str(item['function'])
        #         if pred == target:
        #             wfile.write(fn + "\n")
        #             wfile.write("Prediction: %i \n" % pred)
        #             wfile.write("Target: %i \n" % target)
        #             wfile.write("=================================================================\n")
        #             count += 1
        self.print_msg("Classification Output saved to %s" % output_path)