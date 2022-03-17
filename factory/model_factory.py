from configs.model_type import ModelType
from configs.task_type import TaskType, Task
from pymodels.classification.graph.gat import GATModel
from pymodels.classification.graph.gcn import GCNModel
from pymodels.classification.graph.ggnn import GGNNModel
from pymodels.classification.sequence.lstmclassifymodel import LSTMClassifyModel
from pymodels.classification.sequence.transformer import TransformerEncoderModel
from pymodels.classification.tfidf.naivebayes import NaiveBayesModel
from pymodels.classification.tfidf.svm import SVMModel
from pymodels.classification.tfidf.xgboost import XGBoostModel
from configs.experiment_mode import ExperimentMode
from pymodels.classification.tree.treelstm import TreeLSTMModel
from pymodels.clone_detection.graph.pairwise_gat import PairwiseGATModel
from pymodels.clone_detection.graph.pairwise_gcn import PairwiseGCNModel
from pymodels.clone_detection.graph.pairwise_ggnn import PairwiseGGNNModel
from pymodels.clone_detection.sequence.pairwise_lstm_model import PairwiseLSTMModel
from pymodels.clone_detection.sequence.pairwise_transformer import PairwiseTransformerModel
from pymodels.clone_detection.tree.pairwise_treelstm import PairwiseTreeLSTMModel
from pymodels.summarization.double_vocab_sequence.lstm_summarize_double_vocab_model import LSTMSummarizeDoubleVocabModel
from pymodels.summarization.single_vocab_graph.gat_summarize import GATSummarizeModel
from pymodels.summarization.single_vocab_graph.gcn_summarize import GCNSummarizeModel
from pymodels.summarization.single_vocab_sequence.lstm_summarize_model import LSTMSummarizeModel
from pymodels.summarization.single_vocab_sequence.transformer_summarize_model import \
    TransformerSummarizeSingleVocabModel
from pymodels.summarization.single_vocab_tree.treelstm_summarize import TreeLSTMSummarizeModel


class ModelFactory:
    @staticmethod
    def get_model(config):
        if config.task_type == TaskType.Classification and config.task in [Task.VulnerabilityDetection,
                                                                           Task.CodeClassification,
                                                                           Task.PatchIdentification]:
            if config.model_type in [ModelType.XGBoost]:
                return XGBoostModel
            elif config.model_type in [ModelType.SVM]:
                return SVMModel
            elif config.model_type in [ModelType.NaiveBayes]:
                return NaiveBayesModel
            elif config.model_type in [ModelType.LSTM,
                                       ModelType.BiLSTM]:
                return LSTMClassifyModel
            elif config.model_type == ModelType.TRANSFORMERENCODER:
                return TransformerEncoderModel
            elif config.model_type == ModelType.TreeLSTM:
                return TreeLSTMModel
            elif config.model_type == ModelType.GCN:
                return GCNModel
            elif config.model_type == ModelType.GAT:
                return GATModel
            elif config.model_type == ModelType.GGNN:
                return GGNNModel
            else:
                raise SystemExit(NotImplementedError("Unknown Classification PyModel: %s" % config.experiment_mode))
        # elif config.task_type == TaskType.Classification and config.task == Task.PatchIdentification:
        #     return LSTMClassifyModel

        elif config.task_type == TaskType.PairwiseClassification:
            if config.model_type in [ModelType.XGBoost]:
                return XGBoostModel
            elif config.model_type in [ModelType.SVM]:
                return SVMModel
            elif config.model_type in [ModelType.NaiveBayes]:
                return NaiveBayesModel
            elif config.model_type in [ModelType.LSTM,
                                       ModelType.BiLSTM]:
                return PairwiseLSTMModel
            elif config.model_type == ModelType.TRANSFORMERENCODER:
                return PairwiseTransformerModel
            elif config.model_type == ModelType.TreeLSTM:
                return PairwiseTreeLSTMModel
            elif config.model_type == ModelType.GCN:
                return PairwiseGCNModel
            elif config.model_type == ModelType.GAT:
                return PairwiseGATModel
            elif config.model_type == ModelType.GGNN:
                return PairwiseGGNNModel
            else:
                raise SystemExit(NotImplementedError("Unknown Pairwise Classification PyModel: %s" % config.model_type))



class DummyModel:
    def __init__(self):
        pass
