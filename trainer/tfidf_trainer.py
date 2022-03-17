import time
import datetime

from configs.experiment_mode import ExperimentMode
from configs.task_type import TaskType
from factory.model_factory import ModelFactory
from utils.util import get_pretty_metric, print_msg, is_tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
from evaluation.evaluator.classification_evaluator import ClassificationEvaluator


class TFIDFTrainer:
    def __init__(self, config):
        self.name = "TFIDFTrainer"
        self.config = config
        self.tfidf_vectorizer = None
        self.timestamp = self.config.timestamp
        self.evaluator = ClassificationEvaluator(config)
        self.model_class = ModelFactory().get_model(self.config)
        assert self.model_class is not None, "Model Factory fails to get Model Class"
        self.model = self.model_class(self.config)

    def setup_model(self):
        pass

    def start_train(self, dataset):
        """
        Training Iteration/Epoch
        :param dataset: Dataset Object for Training and Validating
        :return: NIL
        """
        start = time.time()
        train_datapoints = dataset.train_datapoints
        val_datapoints = dataset.val_datapoints
        test_datapoints = dataset.test_datapoints
        if self.tfidf_vectorizer is None:
            self._get_tfidf_vectorizer(train_datapoints)
        self.train(train_datapoints)
        self.config.logger.info("=" * 100)
        self.validate(val_datapoints, "val")
        self.validate(test_datapoints, "test")
        end = time.time()
        total_time = end - start
        self.config.logger.info("Total Time: %s" % str(datetime.timedelta(seconds=total_time)).split(".")[0])

    def train(self, train_datapoints):
        """
        Train self.model using train_datapoints
        :param train_datapoints: Train DataPoints
        :return: Return None as training has no accuracy
        """
        if self.is_pairwise():
            train_x = [dp.function_one + " " + dp.function_two for dp in train_datapoints]
            train_x = self.tfidf_vectorizer.transform(train_x)
        else:
            train_x = self.tfidf_vectorizer.transform([dp.function for dp in train_datapoints])
        train_y = [dp.tgt for dp in train_datapoints]
        self.model.train(train_x, train_y)
        rm_str = "[{:^5}] ".format("train")
        self.config.logger.info(rm_str)
        return None

    def validate(self, val_datapoints, running_mode=""):
        """
        Validate self.model using val_datapoints
        :param val_datapoints: Validation Datapoints
        :param running_mode: Running Mode
        :return: Return the scores
        """
        if self.is_pairwise():
            val_x = [dp.function_one + " " + dp.function_two for dp in val_datapoints]
            val_x = self.tfidf_vectorizer.transform(val_x)
        else:
            val_x = self.tfidf_vectorizer.transform([dp.function for dp in val_datapoints])
        val_y = [dp.tgt for dp in val_datapoints]
        preds = self.model.val(val_x, val_y)
        self.evaluator.add_metric_data(preds, val_y)
        metric_str = self.evaluator.get_pretty_metric(self.evaluator.evaluate_score())
        rm_str = "[{:^5}] ".format(running_mode)
        self.config.logger.info(rm_str + metric_str)

    def _get_tfidf_vectorizer(self, train_datapoints):
        """
        Get TF-IDF Vectorizer based on Training DL
        :param train_datapoints: Trainer DL
        :return:
        """
        vectorizer = TfidfVectorizer()
        if self.is_pairwise():
            corpus = [dp.function_one + " " + dp.function_two for dp in train_datapoints]
        else:
            corpus = [dp.function for dp in train_datapoints]
        vec = vectorizer.fit(corpus)
        self.tfidf_vectorizer = vec

    def is_pairwise(self):
        """
        Check if the classification is pairwise classification
        :return:
        """
        exp_mode = self.config.task_type
        if exp_mode in [TaskType.PairwiseClassification]:
            return True
        return False
