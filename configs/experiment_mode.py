from enum import Enum


class ExperimentMode(Enum):
    # Classification Methods

    # TF-IDF Based Methods
    XGBoost_Classify = "XGBoost_Classify"
    SVM_Classify = "SVM_Classify"
    NaiveBayes_Classify = "NaiveBayes_Classify"

    # Sequence-Based Methods
    LSTM_Classify = "LSTM_Classify"
    BiLSTM_Classify = "BiLSTM_Classify"
    TRANSFORMERENCODER_Classify = "TRANSFORMERENCODER_Classify"

    # Tree-Based Method
    TreeLSTM_Classify = "TreeLSTM_Classify"

    # Graph-Based Methods
    GCN_Classify = 'GraphConvNetwork_Classify'
    GAT_Classify = 'GraphAttentionNetwork_Classify'
    GGNN_Classify = 'GatedGraphNeuralNetwork_Classify'

    # Clone Detection
    # TF-IDF Based Methods
    XGBoost_PairwiseClassify = "XGBoost_PairwiseClassify"
    SVM_PairwiseClassify = "SVM_PairwiseClassify"
    NaiveBayes_PairwiseClassify = "NaiveBayes_PairwiseClassify"

    # Sequence-based Methods
    LSTM_PairwiseClassify = "LSTM_PairwiseClassify"
    BiLSTM_PairwiseClassify = "BiLSTM_PairwiseClassify"
    TRANSFORMERENCODER_PairwiseClassify = "TransformerEncoder_PairwiseClassify"

    # Tree-based Methods
    TreeLSTM_PairwiseClassify = "TreeLSTM_PairwiseClassify"

    # Graph-based Methods
    GCN_PairwiseClassify = "GCN_PairwiseClassify"
    GAT_PairwiseClassify = "GAT_PairwiseClassify"
    GGNN_PairwiseClassify = "GGNN_PairwiseClassify"


