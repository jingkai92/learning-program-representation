class SingleEdgeGraphPairwiseDataPoint:
    def __init__(self):
        """
        Create a single data point, which is a single unit of
        item in the dataset. All the attributes will be set
        by DataFormatter
        """
        self.file_index = ""
        self.function_one_graph = ""  # A DGL Graph
        self.function_two_graph = ""  # A DGL Graph
        self.graph_one_size = ""
        self.graph_two_size = ""
        self.tgt = ""  # Target of the Function
        self.tgt_vec = ""  # Target of the Function
