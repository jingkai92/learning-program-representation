import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tokenizer.vocab_dict import TokenizationConstant


def graph_readout(g, readout_option):
    """
    Carry out the readout based on the readout option,
    Can be either SUM, MAX or MEAN
    :param g: Batched Graph
    :param readout_option: SUM, MAX or MEAN
    :return:
    """
    assert readout_option in ['sum', 'max', 'mean']
    if readout_option == "sum":
        feats = dgl.sum_nodes(g, 'h')
    elif readout_option == 'mean':
        feats = dgl.mean_nodes(g, 'h')
    elif readout_option == 'max':
        feats = dgl.max_nodes(g, 'h')
    else:
        raise NotImplementedError
    return feats


def to_cuda(tensor, use_cuda):
    """
    Send the tensor to cuda device if use_cuda is true
    :param tensor: Tensor to be sent to Cuda device
    :param use_cuda: Specify true if using cuda
    :return: Return cuda-tensor if use_cuda is true
    """
    if use_cuda:
        return tensor.cuda()
    return tensor


def compute_loss(loss_fn, logits, target):
    """
    Compute the loss
    :param loss_fn: Loss Function to compute the loss
    :param logits: Logits of the Batch Output
    :param target: Ground Truth of the Batch
    :return: Return the Loss of the batch
    """
    vocab_len = logits.shape[-1]
    output_true = logits.permute(1, 0, 2).reshape(-1, vocab_len)
    target_true = target.transpose(1, 0).reshape(-1)

    # loss inputs = [Seq Len * Batch Size, Vocab Size] and [Seq Len * Batch Size]
    loss = loss_fn(output_true, target_true)
    return loss

def convert_logit_to_indices(logit):
    """
    Softmax the logits
    :param logit:
    :return:
    """
    assert len(logit.shape) == 3
    probs = F.log_softmax(logit, dim=2)
    indices = torch.argmax(probs, dim=2)
    return indices


def get_average(lst):
    """
    Get the average of the lst
    :param lst: List that contains alot of integers/floats
    :return: Return the Average of lst
    """
    if len(lst) == 0:
        return 0.0
    return sum(lst) / float(len(lst))


def create_mask(input_tensor, lens, use_cuda):
    """
    Create a mask for input_tensor
    :param input_tensor: Tensor for the Mask
    :param lens: A list of length that corresponds to input tensor
    :param use_cuda: Specify if you want to use cuda
    :return:
    """
    mask = np.zeros((len(input_tensor), lens))
    for i in range(len(input_tensor)):
        mask[i, :input_tensor[i]] = 1
    if use_cuda:
        return torch.Tensor(mask).cuda()
    return torch.Tensor(mask)


def stack_seq_to_tensor(seq):
    """
    Stack a list of sequences into tensor using Torch.Stack
    :param seq: Sequences to be packed
    :return:
    """
    if not seq:
        return torch.tensor(seq, dtype=torch.long)
    seq = torch.stack([torch.tensor(x,  dtype=torch.long) for x in seq])
    return seq


def pad_to_max(lst, max_len, pad_token=None):
    """
    Padded the Tokenized Integer List to Max Length
    :param lst: Single List of Integer
    :param max_len: Max Length to be pad
    :param pad_token: Pad Token
    :return: Return the Padded Sequences
    """
    tmp = [x for x in lst]
    if pad_token is None:
        pad_token = int(TokenizationConstant.PAD.value)
    assert max_len > 0
    if len(tmp) >= max_len:
        tmp = tmp[:max_len]
    elif len(tmp) < max_len:
        diff = max_len - len(tmp)
        for i in range(diff):
            tmp.append(pad_token)
    return tmp


def pad_list_to_max(lst, max_len, max_lst_length):
    """
    Padded the Tokenized Integer List to Max Length
    :param lst: Single List of Integer
    :param max_len: Max Length to be pad
    :return: Return the Padded Sequences
    """
    assert max_len > 0
    if len(lst) >= max_len:
        lst = lst[:max_len]
    elif len(lst) < max_len:
        diff = max_len - len(lst)
        for i in range(diff):
            lst.append([int(TokenizationConstant.PAD.value)] * max_lst_length)
    return lst


def get_reprs(g, mean_feats, raw_feats, g_repr, node_repr, hidden_fc, cell_fc):
    """
    A Static method for getting decoder hidden, cell state after GNN layers
    :param g: Unbatched Graph
    :param mean_feats: Feats after Graph Aggregate
    :param raw_feats: Raw Feat after GNN Layer
    :param g_repr: Graph Representation nn Module
    :param node_repr: Node Representation nn Module
    :param hidden_fc: Hidden Fully Connected Layer
    :param cell_fc: Cell Fully Connected Layer
    :return:
    """
    # Graph Representation
    graph_repr = F.leaky_relu(g_repr(mean_feats))

    # Node Representation
    goutput = F.leaky_relu(node_repr(raw_feats))
    g.ndata['h'] = goutput
    unbatched_graph = dgl.unbatch(g)
    node_reprs = [item.ndata['h'] for item in unbatched_graph]
    node_reprs = nn.utils.rnn.pad_sequence(node_reprs, batch_first=True)
    hidden = hidden_fc(graph_repr)
    cell = cell_fc(graph_repr)
    return graph_repr, node_reprs, hidden, cell
