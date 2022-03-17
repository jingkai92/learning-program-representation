class SequenceDataPoint:
    def __init__(self):
        """
        Create a single data point, which is a single unit of
        item in the dataset. All the attributes will be set
        by DataFormatter
        """
        self.file_index = ""
        self.function = ""
        self.function_vec = ""  # Function
        self.tgt = ""  # Target of the Function
        self.tgt_vec = ""

