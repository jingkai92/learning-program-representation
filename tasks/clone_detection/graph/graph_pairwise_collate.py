import dgl

from utils.pymodels_util import pad_to_max, stack_seq_to_tensor


def collate_graph_for_pairwise_classification(samples):
    """
    Collate Function for Sequence
    :param samples: Samples for each batch
    :return:
    """
    datapoint_list = samples
    batch_dict = dict()
    # Graphs
    graphs_one = [dp.function_one_graph for dp in datapoint_list]
    batched_graph_one = dgl.batch(graphs_one)
    batch_dict['graphs_one'] = batched_graph_one

    graphs_two = [dp.function_two_graph for dp in datapoint_list]
    batched_graph_two = dgl.batch(graphs_two)
    batch_dict['graphs_two'] = batched_graph_two

    # Target
    tgt = [dp.tgt for dp in datapoint_list]
    batch_dict['tgt'] = tgt
    return batch_dict
