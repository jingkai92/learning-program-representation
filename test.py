import os
import argparse
from configs.config import Config
from configs.experiment_mode import ExperimentMode
from configs.model_type import ModelType
from factory.dataset_factory import DatasetFactory
from factory.trainer_factory import TrainerFactory
from utils.util import print_msg, torch_setup, load_vocab_dict, get_pretty_metric

name = "EmpStudy[Test]"


def main(args):
    torch_setup(args.gpu_id, name)
    config = Config(args.config_path, test_mode=True)
    config.print_params()
    config.class_weight = False
    config.evaluate = args.evaluate
    config.output_path = os.path.join(config.output_path, args.config_path.split(os.sep)[-2])
    vector_type = args.vector_type
    config.setup_vocab_dict()
    test(config, vector_type)


def test(config, vector_type=""):
    """
    Test the model using the config object
    :param config: Configuration Object
    :param vector_type: Type of Get Vectors, can be empty
    :return: Return the object
    """
    config.test_mode = True
    # Retrieve and Format the dataset
    # TF-IDF does not have testing method
    if config.model_type in [ModelType.XGBoost, ModelType.SVM, ModelType.NaiveBayes]:
        return

    dataset = DatasetFactory().get_dataset(config)(config, test_mode=True)
    trainer = TrainerFactory().get_trainer(config)(config)
    print_msg("Using Trainer %s" % trainer.name, name=name)
    print_msg("Using Dataset %s" % dataset.name, name=name)

    # Start the Training
    trainer.setup_model()
    model_path = os.path.join(config.output_path, "model.pt")
    assert os.path.exists(model_path), "Model path %s does not exists" % model_path
    trainer.load_pymodel(model_path)
    if config.evaluate and vector_type == "":
        print_msg("Evaluating Model", name)
        test_scores = trainer.start_evaluate(dataset)
    elif not config.evaluate and vector_type:
        print_msg("Getting Vector Type", name)
        test_scores = trainer.get_vectors(dataset, vector_type=vector_type)
    else:
        print_msg("Getting Test Score", name)
        test_scores = trainer.start_test(dataset)
    print_msg(get_pretty_metric(test_scores), name)
    return test_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--config_path", help="Configuration Path, YML Path", default="")
    parser.add_argument("--evaluate", help="Specify if you want look into each prediction result",
                        action='store_true')
    parser.add_argument("--gpu_id", help="GPU ID", default="0")
    parser.add_argument("--vector_type", default="")
    args = parser.parse_args()
    main(args)
