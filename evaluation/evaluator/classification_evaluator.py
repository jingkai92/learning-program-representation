import os
import numpy as np
from utils.util import print_msg
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


class ClassificationEvaluator:
    def __init__(self, config):
        """
        ClassificationEvaluator that handles the classification metrics,
        such as F1, Acc, Precision and Recall, Weighted Accuracy
        """
        self.config = config
        # Multi-class, Binary or TopKAcc
        self.classify_mode = config.classify_mode
        self.timestamp = self.config.timestamp
        self.name = "ClassificationEvaluator"
        self.preds = []
        self.labels = []

    def evaluate_score(self):
        """
        Evaluate the item in self.model_outputs and self.ground_truths
        :return: A dictionary of Metrics
        """
        assert len(self.labels) == len(self.preds), "Evaluation Error: List not Equal"
        assert len(self.labels) != 0, "Empty List"
        if self.classify_mode == "TopKAccuracy":
            metrics = self._get_topk_accuracy()
        elif self.classify_mode == "Multi-class":
            metrics = self._get_multiclass()
        elif self.classify_mode == "Binary":
            metrics = self._get_binary()
        else:
            raise NotImplementedError
        return metrics

    def get_pretty_metric(self, metrics):
        """
        Pretty print the metric out, only print the mm_vars
        :return:
        """
        mvars = self.config.monitor_vars
        out_str = ""
        if metrics is None:
            return out_str
        for key, value in metrics.items():
            if key not in mvars:
                continue
            if type(value) == str:
                out_str += "{}: {} | ".format(key, value)
            else:
                out_str += "{}: {:.4f} | ".format(key, value)
        return out_str

    def _get_topk_accuracy(self):
        """
        Compute the multi class metric such that each entry in
        self.labels and self.preds is a int
        :return:
        """
        metrics = dict()
        metrics['top1_accuracy'] = 0
        metrics['top3_accuracy'] = 0
        metrics['top5_accuracy'] = 0
        top1_accuracy = []
        top3_accuracy = []
        top5_accuracy = []
        for i in range(len(self.labels)):
            # preds = np.argmax(self.preds[i], axis=-1)
            topn_idx = self.get_topn_from_list(self.preds[i], n=5)
            lbl = self.labels[i]

            if lbl in topn_idx[:1]:
                top1_accuracy.append(1)
            else:
                top1_accuracy.append(0)

            if lbl in topn_idx[:3]:
                top3_accuracy.append(1)
            else:
                top3_accuracy.append(0)

            if lbl in topn_idx[:5]:
                top5_accuracy.append(1)
            else:
                top5_accuracy.append(0)

        metrics['top1_accuracy'] = sum(top1_accuracy) / float(len(top1_accuracy))
        metrics['top3_accuracy'] = sum(top3_accuracy) / float(len(top3_accuracy))
        metrics['top5_accuracy'] = sum(top5_accuracy) / float(len(top5_accuracy))
        if self.config.creport:
            all_preds = [np.argmax(x, axis=-1) for x in self.preds]
            print(classification_report(self.labels, all_preds))
        self.reset_list()
        return metrics

    def _get_multiclass(self):
        """
        Get the multi-class classification metric, Macro/Weighted Accuracy
        Default will be weighted accuracy
        :return: Return a dictionary with the metric in it.
        """
        metrics = dict()
        best_preds = np.asarray([np.argmax(line) for line in self.preds])
        balanced_acc = balanced_accuracy_score(self.labels, best_preds)
        # print("Accuracy:", balanced_acc)
        metrics['weighted_accuracy'] = balanced_acc
        if self.config.test_mode:
            print(classification_report(self.labels, best_preds))
        self.reset_list()
        return metrics

    def _get_binary(self):
        """
        Get binary score e.g., accuracy, f1, precision and recall
        :return:
        """
        metrics = dict()
        best_preds = np.asarray([np.argmax(line) for line in self.preds])
        acc = accuracy_score(self.labels, best_preds)
        recall = recall_score(self.labels, best_preds)
        prec = precision_score(self.labels, best_preds)
        f1 = f1_score(self.labels, best_preds)
        # print("Accuracy:", balanced_acc)
        metrics['accuracy'] = acc
        metrics['recall'] = recall
        metrics['precision'] = prec
        metrics['f1'] = f1
        positive_count = 0
        negative_count = 0
        for item in best_preds:
            if item == 0:
                negative_count += 1
            if item == 1:
                positive_count += 1
        metrics['positive_count'] = "%s/%s" % (str(positive_count), str(len(best_preds)))
        metrics['negative_count'] = "%s/%s" % (str(negative_count), str(len(best_preds)))
        self.reset_list()
        return metrics

    @staticmethod
    def _save_list_to_txt(item_list, output_path):
        """
        Save the item in the item_list to output_path, with each item as a line
        :param item_list: A list of Items
        :param output_path: Path of the Output
        :return:
        """
        with open(output_path, 'w') as wfile:
            for item in item_list:
                wfile.write("%s\n" % item)

    def reset_list(self):
        """
        Reset the list for each epoch
        :return: NIL
        """
        self.preds = []
        self.labels = []

    def add_metric_data(self, pred, label):
        """
        Add the model item into our list to keep track of the items
        :param pred: Prediction Result of the Model
        :param label: Ground Truth
        :return: NIL
        """
        self.preds.extend(pred)
        self.labels.extend(label)

    def print_msg(self, msg):
        """
        Print the msg
        :param msg:
        :return:
        """
        print_msg(msg, self.name)

    @staticmethod
    def get_topn_from_list(iter_lst, n=5):
        """
        Get the top n indices from the list
        :param n:
        :param iter_lst:
        :return:
        """
        idx_sorted = np.argsort(iter_lst).tolist()
        idx_sorted.reverse()
        return idx_sorted[:n]
