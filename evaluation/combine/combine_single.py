import time
import torch
import lucene
import argparse
from tqdm import tqdm
from evaluation.combine.combine_util import setup_combine_config, setup_localize_trainer, \
    setup_gen_trainer
from utils.util import print_msg, torch_setup
from retrieval.find_top import find_top
from format.collate.collate_functions import _pad_to_max, pack_seq
from config.base.experiment_util import get_experiment_enum
from format.dataloader.bugfixescombine_dataloader import BugFixesCombineDataloader
lucene.initVM()

name = "BugFixesSingleCombine"


def main(args):
    torch_setup(args.gpu_id, name)
    config = setup_combine_config(args.project_type, args.mode, single=False)
    exp_type = get_experiment_enum(args.experiment_type, config)
    dl_obj = BugFixesCombineDataloader(config, exp_type, True,
                                       gen_legacy=config.gen_legacy, loc_legacy=config.gen_legacy)

    K = int(args.k)
    print_msg("TopK=%i" % K, name)
    print_msg("Non-Batch Combine, Project: %s, Mode: %s" % (args.project_type, args.mode), name)
    start_time = time.time()

    # Loading the necessary item for the specified project
    dl_obj.set_ltokenizer(config.localization_model_path,
                          get_experiment_enum(config.localization_model_type, None))
    if args.mode not in ['SEQR']:
        dl_obj.set_gtokenizer(config.generation_model_path,
                              get_experiment_enum(config.generation_model_type, None))
    print_msg("Loading Testing Dataloader", name)
    test_dl = dl_obj.get_testing_dl()

    # Setup Localization Trainer
    config.update_common_dims(config.localization_dims)
    loc_trainer, classify_evaluator = setup_localize_trainer(config, dl_obj)

    # Setup Generation Trainer
    config.update_common_dims(config.generation_dims)
    gen_trainer, translate_evaluator = setup_gen_trainer(config, dl_obj)

    # Perform Localization + Patch Generation
    print_msg("Generating Patches for Testing Items: %i" % len(test_dl), name)
    topk_raccs = []
    for iter, (batch_dict, dp_list) in tqdm(enumerate(test_dl), disable=False):
        localize(loc_trainer, classify_evaluator, (batch_dict, dp_list), k=K)
        # We generate sample by sample
        for dp in dp_list:
            found = generate(gen_trainer, translate_evaluator, dl_obj, dp, k=K, project_type=args.project_type)
            topk_raccs.append(found)
    loc_scores = classify_evaluator.evaluate_score(name="test")
    print_msg("FL-Testing Results -%s" % loc_trainer.get_pretty_score_dict(loc_scores), name=name)

    gen_scores = translate_evaluator.evaluate_score(name='test')
    print_msg("Combined Testing Results -%s" % gen_trainer.get_pretty_score_dict(gen_scores), name=name)
    print("TopK Repair-Accuracy:", sum(topk_raccs) / float(len(topk_raccs)))

    # generate(gen_trainer, translate_evaluator, dl_obj, localized_test_dl, k=K, project_type=args.project_type)

    # if args.mode == "SEQR":
    #     seqr_generate(localized_test_dl, best_config)
    end_time = time.time()
    print_msg("Time for Combine.py: %s" % str(end_time - start_time), name)


def predict(batch_dict, dl_obj, gtrainer, dp, translate_evaluator):
    start_time = time.time()
    model_input, model_output, loss = gtrainer.model(batch_dict, 'test', loss_fn=None)
    ground_truth = [dp.fixed_line]
    model_output = gtrainer._convert_seq_to_wordstr(model_output, dl_obj.token_tokenizer)
    src_string = gtrainer._convert_seq_to_wordstr(model_input, dl_obj.token_tokenizer)
    translate_evaluator.add_strings(src_string, model_output, ground_truth)
    end_time = time.time()
    # print_msg("Time for Single Prediction: %s" % str(end_time - start_time), name)
    return model_output


def localize(loc_trainer, classify_evaluator, iter_items, k=5):
    """
    Localize the DataPoint in Test_DL
    :param loc_trainer: Localize Trainer
    :param classify_evaluator: Evaluator for Classification
    :param iter_items: Item in a Single iterations
    :param k: TopK
    :return: Return the test_dl with localized buggy lines
    """
    # Get LocalizationTrainer
    batch_dict, dp_list = iter_items
    probs, labels, loss = loc_trainer.model(batch_dict, "test", loss_fn=None)
    classify_evaluator.add_metric_data(probs.cpu().tolist(), labels)
    if probs.shape[-1] < k:
        value, idx_srt = torch.topk(probs, k=probs.shape[-1], dim=1)
    else:
        value, idx_srt = torch.topk(probs, k=k, dim=1)
    idx_srt = idx_srt.cpu().tolist()
    for i, dp in enumerate(dp_list):
        sorted_idxs = idx_srt[i]
        topn_list = []
        topn_index = []
        for x in range(k):
            if x >= len(sorted_idxs):
                topn_list.append("")
                topn_index.append(-1)
                continue
            cur_idx = sorted_idxs[x]
            if cur_idx >= len(dp.buggy_context_tk_list):
                topn_list.append("")
                topn_index.append(-1)
            else:
                topn_list.append(dp.buggy_context_tk_list[cur_idx])
                topn_index.append(cur_idx)
        dp.topn_list = topn_list
        dp.topn_index = topn_index


def generate(gen_trainer, translate_evaluator, dl_obj, datapoint, k=5, project_type=""):
    """
    Generate the Patch for Test_DL
    :param gen_trainer: Generate Trainer
    :param translate_evaluator: Evaluator for Classification
    :param dl_obj: DataLoader Object
    :param k: TopK
    :param project_type: Type of the Project, C56974 or DeepFix
    :param datapoint: A Single Datapoint object
    :return: Return topk accuracy for a single patch
    """
    fixed_line = datapoint.fixed_line
    model_outputs = []
    for item in datapoint.topn_list:
        batch_dict, closest = get_datapoint_batch_dict([item], dl_obj, project_type)
        model_output = batch_predict(batch_dict, dl_obj, gen_trainer, [fixed_line], translate_evaluator)
        model_outputs.extend(model_output)

    if fixed_line in model_outputs:
        return 1
    else:
        return 0


def get_datapoint_batch_dict(items, dl_obj, project_type):
    """
    Get Batch Dictionary for Single Object
    :param items: List of Line of the Localized Model
    :param dl_obj: Dataloader Object
    :param project_type: Type of the Project, C56974 or DeepFix
    :return: Return the batch dictionary
    """
    closest = [get_closest(item, project_type) for item in items]
    cfls, cfl_lens = tokenize_list(closest, dl_obj)
    tk_items, tk_item_lens = tokenize_list(items, dl_obj)
    batch_dict = dict()
    batch_dict['buggy_line_len'] = tk_item_lens
    batch_dict['max_buggy_line_len'] = max(batch_dict['buggy_line_len'])
    src_ts = [_pad_to_max(tk_item, batch_dict['max_buggy_line_len']) for tk_item in tk_items]
    batch_dict['buggy_line_tensor'] = pack_seq(src_ts, use_cuda=True)

    # Retrieval
    batch_dict["bline2fline_top1_len"] = cfl_lens
    batch_dict['max_bline2fline_top1_len'] = max(batch_dict["bline2fline_top1_len"])
    retrieval_key_ts = [_pad_to_max(cfl, batch_dict["max_bline2fline_top1_len"]) for cfl in cfls]
    batch_dict['bline2fline_top1_tensor'] = pack_seq(retrieval_key_ts, use_cuda=True)
    return batch_dict, closest


def batch_predict(batch_dict, dl_obj, gtrainer, ground_truth, translate_evaluator):
    model_input, model_output, loss = gtrainer.model(batch_dict, 'test', loss_fn=None)
    model_output = gtrainer._convert_seq_to_wordstr(model_output, dl_obj.token_tokenizer)
    src_string = gtrainer._convert_seq_to_wordstr(model_input, dl_obj.token_tokenizer)
    translate_evaluator.add_strings(src_string, model_output, ground_truth)
    return model_output


def get_closest(code_line, project_type):
    """
    Get the closest line
    :param code_line: A single line of code
    :param project_type: Type of the Project, C56974 or DeepFix
    :return: return a single line of closest line
    """
    if code_line == "":
        return ""
    idx_path = "./retrieval/%s/lucene_index_bline2fline" % project_type.lower()
    closest_line = find_top(code_line, idx_path)
    if closest_line == None:
        closest_line = ""
    return closest_line


def tokenize_list(item_list, dl_obj,):
    """
    Tokenize a list of item with the tokenizer
    :param item_list: Item List
    :param dl_obj: Dataloader object
    :return: Return Tokenized item and its lenght list
    """
    tk_lists = []
    tk_lens = []
    for item in item_list:
        tk_list, tk_len = dl_obj.tokenize_with_gtokenizer(item)
        tk_lists.append(tk_list)
        tk_lens.append(tk_len)
    return tk_lists, tk_lens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-et', '--experiment_type',
                        help="Specify the experiment type etc. See README.md for more information",
                        default="NAIVE")
    parser.add_argument('-p', '--project_type', help="Specify the Project/Dataset etc."
                                                     " See README.md for more information",
                        default="tufano")
    parser.add_argument("-m", "--mode", help="", required=True)
    parser.add_argument('-g', '--gpu_id', help="Specify GPU ID", default="1")
    parser.add_argument('-k', help="Specify Value of K", default=1)
    args = parser.parse_args()
    main(args)
