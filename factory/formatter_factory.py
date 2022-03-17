from configs.experiment_mode import ExperimentMode
from configs.model_type import ModelType
from configs.task_type import TaskType, Task
from tasks.clone_detection.graph.megraph_pairwise_formatter import MultiEdgeGraphPairwiseFormatter
from tasks.clone_detection.graph.segraph_pairwise_formatter import SingleEdgeGraphPairwiseFormatter
from tasks.clone_detection.sequence.sequence_pairwise_formatter import SequencePairwiseFormatter
from tasks.clone_detection.tfidf.tfidf_pairwise_formatter import TFIDFPairwiseFormatter
from tasks.clone_detection.tree.treelstm_pairwise_formatter import TreeLSTMPairwiseFormatter
from tasks.common.graph.megraph_formatter import MultiEdgeGraphFormatter
from tasks.common.graph.segraph_formatter import SingleEdgeGraphFormatter
from tasks.common.graph.treelstm_formatter import TreeLSTMFormatter
from tasks.common.sequence.sequence_formatter import SequenceFormatter
from tasks.common.tfidf.tfidf_formatter import TFIDFFormatter


class FormatterFactory:
    @staticmethod
    def get_formatter(config):
        # Classification Task for Code Classification and Vulnerability Detection
        # Both Tasks uses code so they can use the same formatter
        if config.task_type == TaskType.Classification and config.task in [Task.CodeClassification,
                                                                           Task.VulnerabilityDetection]:
            if config.model_type in [ModelType.NaiveBayes,
                                     ModelType.XGBoost,
                                     ModelType.SVM]:
                return TFIDFFormatter(config)
            elif config.model_type in [ModelType.LSTM,
                                       ModelType.BiLSTM,
                                       ModelType.TRANSFORMERENCODER]:
                return SequenceFormatter(config)
            elif config.model_type == ModelType.TreeLSTM:
                return TreeLSTMFormatter(config)
            elif config.model_type in [ModelType.GCN, ModelType.GAT]:
                return SingleEdgeGraphFormatter(config)
            elif config.model_type == ModelType.GGNN:
                return MultiEdgeGraphFormatter(config)
        elif config.task_type == TaskType.Classification and config.task in [Task.PatchIdentification]:
            if config.model_type in [ModelType.LSTM,
                                     ModelType.BiLSTM,
                                     ModelType.TRANSFORMERENCODER]:
                return SequencePiFormatter(config)
        # Pairwise Classification
        elif config.task_type == TaskType.PairwiseClassification:
            if config.model_type in [ModelType.NaiveBayes,
                                     ModelType.XGBoost,
                                     ModelType.SVM]:
                return TFIDFPairwiseFormatter(config)
            elif config.model_type in [ModelType.LSTM, ModelType.BiLSTM,
                                       ModelType.TRANSFORMERENCODER]:
                return SequencePairwiseFormatter(config)
            elif config.model_type in [ModelType.TreeLSTM]:
                return TreeLSTMPairwiseFormatter(config)
            elif config.model_type in [ModelType.GCN, ModelType.GAT]:
                return SingleEdgeGraphPairwiseFormatter(config)
            elif config.model_type in [ModelType.GGNN]:
                return MultiEdgeGraphPairwiseFormatter(config)
