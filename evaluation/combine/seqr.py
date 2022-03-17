from data_processing.seqr.evaluate_seqr import compute_metric
from tokenizer.code_tokenizer import CodeTokenizer
from utils.util import load_txt


t3_parser = CodeTokenizer(data=[], lang="C", tlevel='t3')


def evaluate(gt_path, pred_path, k=5, beam_size=1):
    """
    Evaluate the prediction with the ground truth.
    The default is n=top5 plausible patches
    :param gt_path: Path of Ground Truth
    :param pred_path: Path of Prediction Results
    :param k: 5
    :param beam_size: Beam Size of Seqr
    :return:
    """
    tgt_items = load_txt(gt_path).splitlines()
    pred_items = load_txt(pred_path).splitlines()
    tgt_items = [x.strip() for x in tgt_items]
    pred_items = [x.strip() for x in pred_items]
    print("Total Number of Statements: %i" % len(tgt_items))
    assert len(pred_items) == (len(tgt_items) * beam_size * k)

    # Number of Prediction per sample is beam size * number of patches
    num_of_preds = beam_size * k
    # grouped_preds = split_by_k(pred_items, num_of_preds)
    metrics, racc = compute_metric(pred_items, tgt_items, num_of_preds)
    return metrics, racc


def split_by_k(iter_list, k):
    """
    Split the list into a list of list of k items
    :param iter_list: A 1-D List
    :param k: Number of element in the second dimensions
    :return: Return a list of list of k items
    """
    grouped_items = []
    for i in range(0, len(iter_list), k):
        grouped_items.append(iter_list[i:i + k])
    return grouped_items


def add_bug_tags(line_list, buggy_loc_num):
    """
    Add the <START_BUG> and <END_BUG>
    :param line_list: Line List
    :param buggy_loc_num: Location/Line Number of the Bug
    :return:
    """
    line_list.insert(buggy_loc_num, "<START_BUG>")
    line_list.insert(buggy_loc_num + 2, "<END_BUG>")
    # line_list[buggy_loc_num] = "<START_BUG>\n " + line_list[buggy_loc_num] + " \n<END_BUG>"
    tok_list = []
    for line in line_list:
        if line == "<START_BUG>" or line == "<END_BUG>":
            tok_list.append(line)
            continue
        tok_line = t3_parser.tokenize(line)
        tok_list.append(tok_line)
    toked_buggy_func = " ".join(tok_list)
    if len(toked_buggy_func.split()) <= 1000:
        return toked_buggy_func
    else:
        if buggy_loc_num == 0:
            tokens = toked_buggy_func.split()
            return " ".join(tokens[:1000])
        else:
            all_tokens = []
            tokens = toked_buggy_func.split()
            start_index = tokens.index("<START_BUG>")
            end_index = tokens.index("<END_BUG>")
            bug_token_count = end_index - start_index - 1 + 2
            onethird = (1000 - bug_token_count) // 3
            if (end_index + onethird) > len(tokens):
                end_tokens = tokens[end_index + 1: len(tokens)]
                remaining_count = 1000 - bug_token_count - len(end_tokens)
                start_tokens = tokens[start_index - remaining_count:start_index]
            else:
                end_tokens = tokens[end_index + 1: end_index + onethird]
                remaining_count = 1000 - bug_token_count - len(end_tokens)
                start_token_index = start_index - (onethird * 2)
                if start_token_index < 0:
                    start_token_index = 0
                start_tokens = tokens[start_token_index:start_index]

            all_tokens.extend(start_tokens)
            all_tokens.extend(tokens[start_index:end_index])
            all_tokens.extend(end_tokens)
            return " ".join(all_tokens)
