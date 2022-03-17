import os
from tqdm import tqdm
from bases.base_dataset import BaseDataset
from configs.experiment_mode import ExperimentMode


class Devign(BaseDataset):
    def __init__(self, config, name="DevignDataset", running='train'):
        """
        Initiate the Dataset of Devign Dataset
        :param config: Configuration file
        """
        BaseDataset.__init__(self, name, config, running)

