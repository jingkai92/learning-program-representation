#!/usr/bin/env python

"""
Usage:
    train.py
    train.py (--config_path FILE) [--gpu_id GPU_ID]
    train.py (--config_pathFILE)
Options:
    -h --help               Show this screen.
    --gpu_id GPUID          GPU ID [default: 0]
    --config_path=FILE      Configuration Path of YML, most likely in ./yml [default: "."]
    --quiet                 Less output (not one per line per minibatch). [default: False]
"""

from test import test
from docopt import docopt
from configs.config import Config
from factory.dataset_factory import DatasetFactory
from factory.trainer_factory import TrainerFactory
from utils.util import *


def main(arguments):
    """
    Entry Method for Code Intelligent Tasks
    :param arguments: Arguments from docopt
    :return: NIL
    """
    # Setup Configuration Object
    config_path = arguments.get('--config_path')
    config = Config(config_path)
    config.print_params()
    config.setup_vocab_dict()

    # Formatting the dataset and start the trainer
    dataset = DatasetFactory().get_dataset(config)
    trainer = TrainerFactory().get_trainer(config)
    config.logger.info("Trainer: %s | Dataset: %s" % (trainer.name, dataset.name))

    # Start the Training
    trainer.setup_model()
    trainer.start_train(dataset)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
