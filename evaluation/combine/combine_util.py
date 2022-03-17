import os

from config.base.experiment_util import get_config, get_experiment_enum
from evaluation.evaluator.classification_evaluator import ClassificationEvaluator
from evaluation.evaluator.translation_evaluator import TranslationEvaluator
from factory.trainer_factory import TrainerFactory
from utils.util import load_json, print_msg


def load_best_config(project_type, mode, context=None):
    """
    Load the best model path from the json file
    :param project_type: Type of the Project, C56974 or DeepFix
    :param mode: Mode of Combination
    :return: Return a Dictionary of best config
    """
    if not context:
        best_folder = "./config/%s_best" % project_type.lower()
        json_path = os.path.join(best_folder, "%s.json" % mode.lower())
    else:
        best_folder = "./config/%s_best" % project_type.lower()
        json_path = os.path.join(best_folder, "%s-%s.json" % (mode.lower(), context))
    print_msg("Loading Best Config JSON from: %s" % json_path, "")
    return load_json(json_path)


def setup_combine_config(project_type, mode, single=False):
    """
    Intermediate Function to setup the configuration for Combining
    :param project_type: Type of the Project
    :param mode: Mode: AutoFLR, DeepFix, etc
    :param single: Single mode
    :return: Return the config object
    """
    config = get_config(project_type)
    if single:
        config.batch_size = 1
    best_config = load_best_config(project_type, mode)

    # Set up Combine Configuration
    # Localization Setting
    config.retrieval_key = best_config['retrieval_key']
    config.gen_legacy = best_config['generation_legacy'] if 'generation_legacy' in best_config else False
    config.loc_legacy = best_config['localization_legacy'] if 'localization_legacy' in best_config else False
    config.localization_model_path = best_config['localization_model_path']
    config.generation_model_path = best_config['generation_model_path']
    config.localization_model_type = best_config['localization_model_type']
    config.generation_model_type = best_config['generation_model_type']
    config.localization_dims = best_config['localization_dimension']

    # Generation Setting
    config.generation_dims = best_config['generation_dimension']
    if 'strong_pos' in best_config:
        config.strong_pos_enc = best_config['strong_pos']
    if config.gen_legacy:
        config.legacy_dim_feedforward = best_config['generation_dim_feedforward']
        config.legacy_nhead = best_config['generation_nlayer']
        config.legacy_trans_dec_n_layers = best_config['generation_nlayer']
        config.legacy_trans_enc_n_layers = best_config['generation_nlayer']
    return config


def setup_localize_trainer(config, dataloader_object):
    """
    Set up the localization trainer for prediction
    :param config: Config Object
    :param dataloader_object: Dataloader object
    :return: Return evaluator and trainer
    """
    model_path = os.path.join(config.localization_model_path, 'model.pt')
    print_msg("FL Model Path: %s" % model_path, "LocalizeTrainerSetup")
    classify_evaluator = ClassificationEvaluator(config, config.output_dir)
    dataloader_object.token_tokenizer = dataloader_object.ltoken_tokenizer
    ltrainer = TrainerFactory().get_trainer(config, dataloader_object,
                                            get_experiment_enum(config.localization_model_type, None))
    ltrainer.legacy = config.loc_legacy
    ltrainer.setup_model()
    ltrainer.load_pymodel(model_path)
    ltrainer.model.eval()
    return ltrainer, classify_evaluator


def setup_gen_trainer(config, dataloader_object):
    """
    Set up the Generation trainer for prediction
    :param config: Config Object
    :param dataloader_object: Dataloader object
    :return: Return evaluator and trainer
    """
    model_path = os.path.join(config.generation_model_path, 'model.pt')
    print_msg("PG Model Path: %s" % model_path, "GenerateTrainerSetup")
    translate_evaluator = TranslationEvaluator(config, config.output_dir)
    dataloader_object.token_tokenizer = dataloader_object.gtoken_tokenizer
    gtrainer = TrainerFactory().get_trainer(config, dataloader_object,
                                            get_experiment_enum(config.generation_model_type, None))
    gtrainer.legacy = config.gen_legacy
    if gtrainer.legacy:
        config.dim_feedforward = config.legacy_dim_feedforward
        config.nhead = config.legacy_nhead
        config.trans_dec_n_layers = config.legacy_trans_dec_n_layers
        config.trans_enc_n_layers = config.legacy_trans_enc_n_layers
    gtrainer.setup_model()
    gtrainer.load_pymodel(model_path)
    gtrainer.model.eval()
    return gtrainer, translate_evaluator