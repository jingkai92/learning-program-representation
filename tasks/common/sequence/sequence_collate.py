from utils.pymodels_util import stack_seq_to_tensor, pad_to_max, pad_list_to_max


def collate_sequence_for_classification(samples):
    """
    Collate Function for Classification
    :param samples: Samples for each batch
    :return:
    """
    datapoint_list = samples
    batch_dict = dict()

    funcs = [dp.function_vec for dp in datapoint_list]
    batch_dict["funcs_lens"] = [len(fn) for fn in funcs]
    largest_funcs_len = max(batch_dict['funcs_lens'])
    fn_ts = [pad_to_max(fn, largest_funcs_len) for fn in funcs]
    batch_dict['fn_tensors'] = stack_seq_to_tensor(fn_ts)
    batch_dict['tgt'] = [dp.tgt for dp in datapoint_list]

    # Function
    batch_dict['function'] = [dp.function for dp in datapoint_list]
    batch_dict['fid'] = [dp.file_index for dp in datapoint_list]

    return batch_dict


def collate_sequence_for_summarization(samples):
    """
    Collate the function for Summarization Tasks
    :param samples:
    :return:
    """
    datapoint_list = samples
    batch_dict = dict()

    funcs = [dp.function_vec for dp in datapoint_list]
    batch_dict["funcs_lens"] = [len(fn) for fn in funcs]
    largest_funcs_len = max(batch_dict['funcs_lens'])
    fn_ts = [pad_to_max(fn, largest_funcs_len) for fn in funcs]
    batch_dict['fn_tensors'] = stack_seq_to_tensor(fn_ts)

    tgts = [dp.tgt_vec for dp in datapoint_list]
    batch_dict["tgt_lens"] = [len(tgt) for tgt in tgts]
    batch_dict["tgt"] = [dp.tgt for dp in datapoint_list]
    largest_tgt_len = max(batch_dict['tgt_lens'])
    tgt_ts = [pad_to_max(t, largest_tgt_len) for t in tgts]
    batch_dict['tgt_tensors'] = stack_seq_to_tensor(tgt_ts)
    return batch_dict


def collate_sequence_for_name_prediction(samples):
    """
    Collate the graphs for name_prediction
    :param samples:
    :return:
    """
    pass