from configs.experiment_mode import ExperimentMode
from configs.model_type import ModelType
from trainer.sequence_trainer import SequenceTrainer
from trainer.summarize_trainer import SummarizeTrainer
from trainer.tfidf_trainer import TFIDFTrainer


class TrainerFactory:
    def __init__(self):
        self.tfidf_experiments = [ModelType.XGBoost,
                                  ModelType.NaiveBayes,
                                  ModelType.SVM]

        self.sequence_experiments = [ModelType.LSTM,
                                     ModelType.BiLSTM,
                                     ModelType.TRANSFORMERENCODER]

        self.graph_experiments = [ModelType.TreeLSTM,
                                  ModelType.GCN,
                                  ModelType.GAT,
                                  ModelType.GGNN,
                                  ExperimentMode.TreeLSTM_Classify,
                                  ExperimentMode.TreeLSTM_PairwiseClassify,
                                  ExperimentMode.GCN_PairwiseClassify,
                                  ExperimentMode.GAT_PairwiseClassify,
                                  ExperimentMode.GGNN_PairwiseClassify]

    def get_trainer(self, config):
        if config.model_type in self.tfidf_experiments:
            return TFIDFTrainer(config)
        elif config.model_type in self.sequence_experiments or config.model_type in self.graph_experiments:
            # We can use SequenceTrainer for Graph Training too.
            return SequenceTrainer(config)
        else:
            raise SystemExit(NotImplementedError("%s not found" % config.experiment_mode))
