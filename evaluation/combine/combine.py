import os
import time
import torch
import lucene
import argparse
import subprocess

import numpy as np
from tqdm import tqdm

from evaluation.combine.combine_util import load_best_config
from utils.util import print_msg, torch_setup, get_pretty_metric
from retrieval.find_top import find_top
from evaluation.combine.seqr import evaluate, add_bug_tags
from utils.util import save_list_to_txt
from factory.trainer_factory import TrainerFactory
from evaluation.evaluator.translation_evaluator import TranslationEvaluator
from format.collate.collate_functions import _pad_to_max, pack_seq
from config.base.experiment_util import get_config, get_experiment_enum
from evaluation.evaluator.classification_evaluator import ClassificationEvaluator
from format.dataloader.bugfixescombine_dataloader import BugFixesCombineDataloader
lucene.initVM()

name = "BugFixesCombine"
N = 5


def main(args):
    torch_setup(args.gpu_id, name)
    config = get_config(args.project_type)
    config.print_params()
    print_msg("N=%i" % N, name)
    print_msg("Conducting Combination Operation for Project: %s in Mode: %s" % (args.project_type, args.mode), name)
    start_time = time.time()
    best_config = load_best_config(args.project_type, args.mode, args.context)
    if 'retrieval_key' in best_config:
        config.retrieval_key = best_config['retrieval_key']
    else:
        config.retrieval_key = ""
    # Loading the necessary item for the specified project
    exp_type = get_experiment_enum(args.experiment_type, config)
    gen_legacy = best_config['generation_legacy'] if 'generation_legacy' in best_config else False
    loc_legacy = best_config['localization_legacy'] if 'localization_legacy' in best_config else False

    dl_obj = BugFixesCombineDataloader(config, exp_type, True, gen_legacy=gen_legacy, loc_legacy=loc_legacy)
    dl_obj.set_ltokenizer(best_config['localization_model_path'], get_experiment_enum(best_config['localization_model_type'], None))
    if args.mode not in ['SEQR_1', "SEQR_5", "SEQR_50"]:
        dl_obj.set_gtokenizer(best_config['generation_model_path'], get_experiment_enum(best_config['generation_model_type'], None))
    print_msg("Loading Testing Dataloader", name)
    test_dl = dl_obj.get_testing_dl()
    localized_test_dl = localize(config, dl_obj, test_dl, best_config)
    # else:
    #     localized_test_dl = perfect_localize(config, dl_obj, test_dl, best_config)

    if args.mode in ['SEQR_1', "SEQR_5", "SEQR_50"]:
        seqr_generate(localized_test_dl, best_config)
    else:
        if 'generation_legacy' in best_config and best_config['generation_legacy'] == 1:
            generate(config, dl_obj, localized_test_dl, best_config, legacy=True, project_type=args.project_type,
                     retrieval_key=config.retrieval_key)
        else:
            if 'strong_pos' in best_config and best_config['strong_pos'] == 1:
                config.strong_pos_enc = True
            generate(config, dl_obj, localized_test_dl, best_config, project_type=args.project_type,
                     retrieval_key=config.retrieval_key)
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


def localize(config, dl_obj, test_dl, best_config):
    """
    Localize the DataPoint in Test_DL
    :param config: Config Object
    :param dl_obj: DataLoader Object
    :param test_dl: Test Dataloder
    :param best_config: best_config
    :return: Return the test_dl with localized buggy lines
    """
    # Get LocalizationTrainer
    print("\n")
    config.update_common_dims(best_config['localization_dimension'])
    print_msg("======= Localization of Bug ========", name)
    classify_evaluator = ClassificationEvaluator(config, config.output_dir)
    dl_obj.token_tokenizer = dl_obj.ltoken_tokenizer
    ltrainer = TrainerFactory().get_trainer(config, dl_obj, get_experiment_enum(best_config['localization_model_type'], None))
    if 'localization_legacy' in best_config and best_config['localization_legacy'] == 1:
        ltrainer.legacy = True
    ltrainer.setup_model()
    model_path = os.path.join(best_config['localization_model_path'], 'model.pt')
    print_msg("Loading Model Path from %s" % model_path, name)
    ltrainer.load_pymodel(model_path)
    ltrainer.model.eval()
    for iter, (batch_dict, dp_list) in tqdm(enumerate(test_dl), disable=False):
        probs, labels, loss = ltrainer.model(batch_dict, "test", loss_fn=None)
        classify_evaluator.add_metric_data(probs.cpu().tolist(), labels)
        value, idx_srt = torch.topk(probs, k=N, dim=1)
        idx_srt = idx_srt.cpu().tolist()

        for i, dp in enumerate(dp_list):
            sorted_idxs = idx_srt[i]
            topn_list = []
            topn_index = []
            for x in range(N):
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

        localized_list = []
        for i, prob in enumerate(probs):
            idx_sorted = np.argsort(prob.cpu().tolist()).tolist()
            idx_sorted.reverse()
            sorted_n = idx_sorted[:N]
            if labels[i] in sorted_n:
                localized_list.append(1)
            else:
                localized_list.append(0)

        for i, dp in enumerate(dp_list):
            if localized_list[i] == 1:
                dp.localized = True
            else:
                dp.localized = False
    scores = classify_evaluator.evaluate_score(name="test")
    print_msg("Testing Results -%s" % ltrainer.get_pretty_score_dict(scores), name=name)
    return test_dl


def perfect_localize(config, dl_obj, test_dl, best_config):
    """
    Localize the DataPoint in Test_DL
    :param config: Config Object
    :param dl_obj: DataLoader Object
    :param test_dl: Test Dataloder
    :param best_config: best_config
    :return: Return the test_dl with localized buggy lines
    """
    # Get LocalizationTrainer
    print("\n")
    # config.update_common_dims(best_config['localization_dimension'])
    print_msg("======= Localization of Bug ========", name)
    # classify_evaluator = ClassificationEvaluator(config, config.output_dir)
    # dl_obj.token_tokenizer = dl_obj.ltoken_tokenizer
    # ltrainer = TrainerFactory().get_trainer(config, dl_obj, get_experiment_enum(best_config['localization_model_type'], None))
    # if 'localization_legacy' in best_config and best_config['localization_legacy'] == 1:
    #     ltrainer.legacy = True
    # ltrainer.setup_model()
    # model_path = os.path.join(best_config['localization_model_path'], 'model.pt')
    # print_msg("Loading Model Path from %s" % model_path, name)
    # ltrainer.load_pymodel(model_path)
    # ltrainer.model.eval()
    for iter, (batch_dict, dp_list) in tqdm(enumerate(test_dl), disable=False):
        labels = batch_dict['buggy_loc']
        for i, dp in enumerate(dp_list):
            # sorted_idxs = idx_srt[i]
            topn_list = []
            topn_index = []
            for x in range(N):
                topn_list.append(dp.buggy_context_tk_list[labels[i]])
                topn_index.append(labels[i])
            dp.topn_list = topn_list
            dp.topn_index = topn_index
    # scores = classify_evaluator.evaluate_score(name="test")
    # print_msg("Testing Results -%s" % ltrainer.get_pretty_score_dict(scores), name=name)
    return test_dl


def generate(config, dl_obj, test_dl, best_config, legacy=False, project_type="", retrieval_key=""):
    """
    Generate the Patch for Test_DL
    :param config: Config Object
    :param project_type: Type of the Project, C56974 or DeepFix
    :param dl_obj: DataLoader Object
    :param test_dl: Test Dataloder
    :param best_config: best_config
    :param legacy: Specify if you want legacy transformer
    :return: Return the test_dl with localized buggy lines
    """
    print("\n")
    config.update_common_dims(best_config['generation_dimension'])
    print_msg("======= Generation of BugFixes (Combined)========", name)
    translate_evaluator = TranslationEvaluator(config, config.output_dir)
    dl_obj.token_tokenizer = dl_obj.gtoken_tokenizer
    gtrainer = TrainerFactory().get_trainer(config, dl_obj, get_experiment_enum(best_config['generation_model_type'], None))
    if legacy:
        # For the best AutoFLR Model
        gtrainer.legacy = legacy
        config.dim_feedforward = best_config['generation_dim_feedforward']
        config.nhead = best_config['generation_nlayer']
        config.trans_dec_n_layers = best_config['generation_nlayer']
        config.trans_enc_n_layers = best_config['generation_nlayer']
    print("Retrieval Key: %s" % config.retrieval_key)
    gtrainer.setup_model()
    model_path = os.path.join(best_config['generation_model_path'], 'model.pt')
    print_msg("Loading Model Path from %s" % model_path, name)
    gtrainer.load_pymodel(model_path)
    gtrainer.model.eval()
    in_top5_output = []
    all_inputs = []
    all_closest = []
    all_outputs = []
    all_ground_truth = []
    all_indexes = []
    tmp_count = 0
    print_msg("Predicting for Testing Items: %i" % (len(test_dl) * config.batch_size), name)
    for iter, (batch_dict, dp_list) in tqdm(enumerate(test_dl), disable=False):
        dp_topn_items = []
        dp_topn_indexes = []
        dp_fixed = []
        loc_ground_truth = []
        idxs = []
        for dp in dp_list:
            fixed_line = dp.fixed_line
            topn_items = [item for item in dp.topn_list]
            dp_topn_indexes.append(dp.topn_index)
            dp_topn_items.extend(topn_items)
            loc_ground_truth.append(dp.buggy_loc)
            for i in range(N):
                dp_fixed.append(fixed_line)
                try:
                    idxs.append(dp.file_index)
                except:
                    pass

        batch_dict, closest = get_batch_dict_for_top5items(dp_topn_items, dl_obj, project_type, retrieval_key)
        model_output = batch_predict(batch_dict, dl_obj, gtrainer, dp_fixed, translate_evaluator)

        # Compute Repair Accuracy
        outputs, fixed = split_to_k_size(model_output, dp_fixed, N=N)
        # print(len(outputs))
        # print(len(fixed))
        # print(outputs[0])
        # print(fixed[0])
        # print(dp_topn_indexes[0])
        # print(loc_ground_truth[0])
        # print(len(dp_list))
        for i in range(len(dp_list)):
            cur_dp = dp_list[i]
            cur_topn_index = dp_topn_indexes[i]
            cur_loc_gt = loc_ground_truth[i]
            cur_model_output = outputs[i]
            cur_fixed = fixed[i]
            found = False
            for i, item in enumerate(cur_model_output):
                if item == cur_fixed[i] and \
                        cur_topn_index[i] == cur_loc_gt:
                    found = True
                    break
            if found:
                in_top5_output.append(1)
            else:
                in_top5_output.append(0)

        # all_outputs.extend(model_output)
        # all_ground_truth.extend(dp_fixed)
        # all_closest.extend(closest)
        # all_inputs.extend(dp_topn_items)
        # all_indexes.extend(idxs)
    scores = translate_evaluator.evaluate_score(name='test')
    print_msg("Testing Results -%s" % gtrainer.get_pretty_score_dict(scores), name=name)
    print("RAcc:", sum(in_top5_output) / float(len(in_top5_output)))
    print("RAcc (Digits): %i/%i" % (sum(in_top5_output), len(in_top5_output)))
    # all_inputs = []
    # all_closest = []
    # all_outputs = []
    # all_ground_truth = []
    # combine_output = os.path.join(config.output_dir, "combine.json")
    # json_list = []
    # for i in range(len(all_inputs)):
    #     json_dict = dict()
    #     # json_dict['index'] = all_indexes[i]
    #     json_dict['input'] = all_inputs[i]
    #     json_dict['output'] = all_outputs[i]
    #     json_dict['ground_truth'] = all_ground_truth[i]
    #     if config.retrieval_key:
    #         json_dict['retrieval_item'] = all_closest[i]
    #     json_list.append(json_dict)
    # save_json(json_list, combine_output)
    # print_msg("Combine JSON saved to %s" % combine_output, name)


def seqr_generate(test_dl, best_config):
    """
    Generate for SEQR is through the OPENNMT.py
    :param test_dl: Localized Test DL
    :return:
    """
    dp_topn_indexes = []
    base_folder = "/home/jingkai/projects/bug-fixes/evaluation/combine/"
    lres_path = os.path.join(base_folder, "seqr_output/src-test.txt")
    ltgt_path = os.path.join(base_folder, "seqr_output/tgt-test.txt")
    output_path = os.path.join(base_folder, "seqr_output/pred-test_beam1.txt")
    lres_list = []
    ltgt_list = []
    for iter, (batch_dict, dp_list) in tqdm(enumerate(test_dl), disable=False):
        # We will input one at a time first
        for dp in dp_list:
            for i, item in enumerate(dp.topn_list):
                buggy_tk_list = dp.buggy_context_tk_list[:]
                if dp.topn_index[i] == -1:
                    lres_list.append(" ".join(buggy_tk_list))
                    continue
                seqr_buggy_line = add_bug_tags(buggy_tk_list, dp.topn_index[i])
                lres_list.append(seqr_buggy_line)
            ltgt_list.append(dp.fixed_line)
    save_list_to_txt(lres_path, lres_list)
    save_list_to_txt(ltgt_path, ltgt_list)
    print_msg("Running SeqR with SubProcess", name=name)
    # Configuration for calling SEQR
    opennmt_path = "/home/jingkai/projects/OpenNMT-py/translate.py"
    gen_path = best_config['generation_model_path']
    cmd = "/home/jingkai/anaconda3/envs/seqr/bin/python %s -model %s -src %s -beam_size %i -n_best %i -output %s -dynamic_dict" % (opennmt_path, gen_path, lres_path, best_config['seqr_beam'], best_config['seqr_beam'], output_path)
    output = subprocess.getoutput(cmd)
    print_msg("Output of SeqR", name=name)
    print(output)
    metrics, acc_list = evaluate(ltgt_path, output_path, k=5, beam_size=best_config['seqr_beam'])
    print(get_pretty_metric(metrics))
    print("RAcc: %f" % (sum(acc_list) / float(len(acc_list))))
    print("Repair Count: [%i/%i]" % (sum(acc_list), len(acc_list)))


def get_batch_dict_for_top5items(items, dl_obj, project_type, retrieval_key):
    """
    Get Batch Dictionary for Single Object
    :param items: List of Line of the Localized Model
    :param dl_obj: Dataloader Object
    :param project_type: Type of the Project, C56974 or DeepFix
    :param retrieval_key: For Context Analysis
    :return: Return the batch dictionary
    """
    closest = []
    if retrieval_key:
        closest = [get_closest(item, project_type, retrieval_key) for item in items]
        cfls, cfl_lens = tokenize_list(closest, dl_obj)

    tk_items, tk_item_lens = tokenize_list(items, dl_obj)
    batch_dict = dict()
    batch_dict['buggy_line_len'] = tk_item_lens
    batch_dict['max_buggy_line_len'] = max(batch_dict['buggy_line_len'])
    src_ts = [_pad_to_max(tk_item, batch_dict['max_buggy_line_len']) for tk_item in tk_items]
    batch_dict['buggy_line_tensor'] = pack_seq(src_ts, use_cuda=True)

    # Retrieval
    if retrieval_key:
        batch_dict["%s_len" % retrieval_key] = cfl_lens
        batch_dict['max_%s_len' % retrieval_key] = max(batch_dict["%s_len" % retrieval_key])
        retrieval_key_ts = [_pad_to_max(cfl, batch_dict['max_%s_len' % retrieval_key]) for cfl in cfls]
        batch_dict['%s_tensor' % retrieval_key] = pack_seq(retrieval_key_ts, use_cuda=True)
    return batch_dict, closest


def batch_predict(batch_dict, dl_obj, gtrainer, ground_truth, translate_evaluator):
    model_input, model_output, loss = gtrainer.model(batch_dict, 'test', loss_fn=None)
    model_output = gtrainer._convert_seq_to_wordstr(model_output, dl_obj.token_tokenizer)
    src_string = gtrainer._convert_seq_to_wordstr(model_input, dl_obj.token_tokenizer)
    translate_evaluator.add_strings(src_string, model_output, ground_truth)
    return model_output


def get_closest(code_line, project_type, retrieval_key):
    """
    Get the closest line
    :param code_line: A single line of code
    :param project_type: Type of the Project, C56974 or DeepFix
    :return: return a single line of closest line
    """
    if code_line == "":
        return ""
    folder_key = "_".join(retrieval_key.split("_")[:-1])
    idx_path = "./retrieval/%s/lucene_index_%s" % (project_type.lower(), folder_key)
    closest_line = find_top(code_line, idx_path, data_type=folder_key)
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


def split_to_k_size(model_output, dp_fixed, N):
    output_in_batch = []
    dp_fixed_in_batch = []
    for i in range(0, len(model_output), N):
        output_in_batch.append(model_output[i:i + N])
        dp_fixed_in_batch.append(dp_fixed[i:i + N])
    assert len(output_in_batch) == len(dp_fixed_in_batch)
    return output_in_batch, dp_fixed_in_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-et', '--experiment_type',
                        help="Specify the experiment type etc. See README.md for more information",
                        default="NAIVE")
    parser.add_argument('-p', '--project_type', help="Specify the Project/Dataset etc."
                                                     " See README.md for more information",
                        default="tufano")
    parser.add_argument("-m", "--mode", help="", required=True)
    parser.add_argument('-g', '--gpu_id', help="Specify GPU ID", default="2")
    parser.add_argument('--context', help="", default="")
    args = parser.parse_args()
    main(args)
