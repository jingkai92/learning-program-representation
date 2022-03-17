from configs.model_type import ModelType
from configs.task_type import TaskType, Task
from tasks.clone_detection.graph.graph_pairwise_collate import collate_graph_for_pairwise_classification
from tasks.clone_detection.sequence.sequence_pairwise_collate import collate_sequence_for_pairwise_classification
from tasks.common.graph.graph_collate import collate_graph_for_classification
from tasks.common.sequence.sequence_collate import collate_sequence_for_classification


class CollateFactory:
    @staticmethod
    def get_collate_fn(config):
        # All TF-IDF methods does not have collate function
        if config.model_type in [ModelType.NaiveBayes,
                                 ModelType.XGBoost,
                                 ModelType.SVM]:
            return None
        # Classification Task
        elif config.task_type == TaskType.Classification and config.task in [Task.VulnerabilityDetection,
                                                                             Task.CodeClassification]:
            if config.model_type in [ModelType.LSTM,
                                     ModelType.BiLSTM,
                                     ModelType.TRANSFORMERENCODER]:
                return collate_sequence_for_classification
            elif config.model_type in [ModelType.TreeLSTM,
                                       ModelType.GCN,
                                       ModelType.GAT,
                                       ModelType.GGNN]:
                return collate_graph_for_classification
            else:
                raise SystemExit(NotImplementedError(
                    "Unknown Collate Fn in CollateFactory for classification" % config.model_type))

        # Pairwise Classification
        elif config.task_type == TaskType.PairwiseClassification:
            if config.model_type in [ModelType.LSTM,
                                     ModelType.BiLSTM,
                                     ModelType.TRANSFORMERENCODER]:
                return collate_sequence_for_pairwise_classification
            elif config.model_type in [ModelType.TreeLSTM, ModelType.GCN,
                                       ModelType.GAT, ModelType.GGNN]:
                return collate_graph_for_pairwise_classification
            else:
                raise SystemExit(NotImplementedError(
                    "Unknown Collate Fn in CollateFactory for pairwise classification: %s" % config.model_type))
