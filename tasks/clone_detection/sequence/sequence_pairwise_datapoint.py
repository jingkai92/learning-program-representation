class SequencePairwiseDataPoint:
    def __init__(self):
        """
        Create a single data point, which is a single unit of
        item in the dataset. All the attributes will be set
        by DataFormatter
        """
        self.file_index = ""
        self.function_one_vec = ""  # Function one
        self.function_two_vec = ""  # Function two
        self.tgt = ""  # Target of the Function
        self.tgt_vec = ""

