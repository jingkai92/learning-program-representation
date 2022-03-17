from bases.base_dataset import BaseDataset


class POJ104Clone(BaseDataset):
    def __init__(self, config, name="POJ104CloneDataset", running_mode='train'):
        """
        Initialize Online Judge Dataset
        :param config: Config Object
        :param running_mode: Specify if you are testing or training
        """
        BaseDataset.__init__(self, name, config, running_mode)


