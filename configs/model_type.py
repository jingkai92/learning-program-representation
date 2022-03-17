from enum import Enum


class ModelType(Enum):
    # TF-IDF Based Methods
    XGBoost = "XGBoost"
    SVM = "SVM"
    NaiveBayes = "NaiveBayes"

    # Sequence-Based Methods
    LSTM = "LSTM"
    BiLSTM = "BiLSTM"
    TRANSFORMERENCODER = "TRANSFORMERENCODER"

    # Tree-Based Method
    TreeLSTM = "TreeLSTM"

    # Graph-Based Methods
    GCN = 'GraphConvNetwork'
    GAT = 'GraphAttentionNetwork'
    GGNN = 'GatedGraphNeuralNetwork'
