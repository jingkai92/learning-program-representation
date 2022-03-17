class GraphDataPoint:
    def __init__(self):
        """
        Create a single data point, which is a single unit of
        item in the dataset. All the attributes will be set
        by DataFormatter
        """
        self.file_index = ""
        self.function = ""
        self.function_graph = ""  # A DGL Graph
        self.graph_size = ""
        self.tgt = ""  # Target of the Function
        self.tgt_vec = ""  # Target of the Function


# class MultiEdgeGraphDataPoint:
#     def __init__(self):
#         """
#         Create a single data point, which is a single unit of
#         item in the dataset. All the attributes will be set
#         by DataFormatter
#         """
#         self.file_index = ""
#         self.function_graph = ""  # A DGL Graph
#         self.graph_size = ""
#         self.edge_types = ""  # A list of type for each edge
#         self.tgt = ""  # Target of the Function
