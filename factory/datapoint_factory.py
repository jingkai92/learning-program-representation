from configs.experiment_mode import ExperimentMode
from configs.model_type import ModelType
from configs.task_type import TaskType
from tasks.clone_detection.graph.graph_pairwise_datapoint import SingleEdgeGraphPairwiseDataPoint
from tasks.clone_detection.sequence.sequence_pairwise_datapoint import SequencePairwiseDataPoint
from tasks.clone_detection.tfidf.tfidf_pairwise_datapoint import TFIDFPairwiseDataPoint
from tasks.common.graph.graph_datapoint import GraphDataPoint
from tasks.common.sequence.sequence_datapoint import SequenceDataPoint
from tasks.common.tfidf.tfidf_datapoint import TFIDFDataPoint


class DataPointFactory:
    @staticmethod
    def get_datapoint(config):
        # Classification Task
        if config.task_type == TaskType.Classification:
            if config.model_type in [ModelType.NaiveBayes,
                                     ModelType.XGBoost,
                                     ModelType.SVM]:
                return TFIDFDataPoint
            elif config.model_type in [ModelType.LSTM,
                                       ModelType.BiLSTM,
                                       ModelType.TRANSFORMERENCODER]:
                return SequenceDataPoint
            elif config.model_type in [ModelType.TreeLSTM,
                                       ModelType.GCN,
                                       ModelType.GAT,
                                       ModelType.GGNN]:
                return GraphDataPoint
            else:
                raise SystemExit(NotImplementedError("Unknown Model Type in Datapoint Factory for classification" % config.model_type))
        # Pairwise Classification
        elif config.task_type == TaskType.PairwiseClassification:
            if config.model_type in [ModelType.NaiveBayes,
                                     ModelType.XGBoost,
                                     ModelType.SVM]:
                return TFIDFPairwiseDataPoint
            elif config.model_type in [ModelType.LSTM, ModelType.BiLSTM,
                                       ModelType.TRANSFORMERENCODER]:
                return SequencePairwiseDataPoint
            elif config.model_type == ModelType.TreeLSTM:
                return TFIDFPairwiseDataPoint
            elif config.model_type in [ModelType.GAT, ModelType.GCN, ModelType.GGNN]:
                return SingleEdgeGraphPairwiseDataPoint
            else:
                raise SystemExit(NotImplementedError("Unknown Model Type in Datapoint Factory for pairwise classification" % config.model_type))

        if config.experiment_mode in [ExperimentMode.XGBoost_Classify, ExperimentMode.SVM_Classify, ExperimentMode.NaiveBayes_Classify]:
            return TFIDFDataPoint
        elif config.experiment_mode in [ExperimentMode.LSTM_Classify, ExperimentMode.BiLSTM_Classify, ExperimentMode.TRANSFORMERENCODER_Classify,
                                        ExperimentMode.LSTM_SummarizeSingleVocab,
                                        ExperimentMode.LSTM_SummarizeDoubleVocab,
                                        ExperimentMode.TRANSFORMERENCODER_SummarizeSingleVocab]:
            return SequenceDataPoint
        elif config.experiment_mode in [ExperimentMode.GCN_Classify, ExperimentMode.GAT_Classify,
                                        ExperimentMode.TreeLSTM_Classify,
                                        ExperimentMode.TreeLSTM_SummarizeSingleVocab,
                                        ExperimentMode.GCN_SummarizeSingleVocab,
                                        ExperimentMode.GAT_SummarizeSingleVocab]:
            return GraphDataPoint
        elif config.experiment_mode == ExperimentMode.GGNN_Classify:
            return GraphDataPoint

        elif config.experiment_mode in [ExperimentMode.SVM_PairwiseClassify,
                                        ExperimentMode.XGBoost_PairwiseClassify,
                                        ExperimentMode.NaiveBayes_PairwiseClassify]:
            return TFIDFPairwiseDataPoint
        elif config.experiment_mode in [ExperimentMode.LSTM_PairwiseClassify,
                                        ExperimentMode.BiLSTM_PairwiseClassify,
                                        ExperimentMode.TRANSFORMERENCODER_PairwiseClassify]:
            return SequencePairwiseDataPoint
        elif config.experiment_mode in [ExperimentMode.TreeLSTM_PairwiseClassify,
                                        ExperimentMode.GCN_PairwiseClassify,
                                        ExperimentMode.GAT_PairwiseClassify,
                                        ExperimentMode.GGNN_PairwiseClassify]:
            return SingleEdgeGraphPairwiseDataPoint
        else:
            raise SystemExit(NotImplementedError("Experiment Type %s is not implemented "
                                                 "in Datapoint" % config.experiment_mode))
