import os
import gzip
import json
import datetime
import torch
import logging
import statistics
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from configs.model_type import ModelType
from tokenizer.vocab_dict import VocabDict, BPEVocabDict

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def load_c_file(c_file_path):
    """
    Load the C File in C File Path as a string
    :param c_file_path: The path of the C File
    :return: Return a string of the C File
    """
    try:
        with open(c_file_path, encoding='utf-8') as rfile:
            code_content = rfile.read()
        return code_content
    except UnicodeDecodeError:
        with open(c_file_path, encoding='windows-1252') as rfile:
            code_content = rfile.read()
        return code_content


def load_txt(path):
    """
    Load a Text File or C File
    :param path: Path of Text File or CFile
    :return:
    """
    ct = None
    with open(path, errors="replace") as rfile:
        ct = rfile.read()
    return ct


def save_list_to_txt(path, lst):
    """
    Save a list to txt
    :param path: Path of Text File or CFile
    :param lst: List to be saved
    :return:
    """
    with open(path, "w") as wfile:
        for item in lst:
            wfile.write("%s\n" % item)
    return


def load_txt_to_list(path):
    """
    Load a list from a .csv path
    :param path: Path of the csv
    :return:
    """
    tmp = []
    with open(path) as rfile:
        for item in rfile:
            tmp.append(item.strip())
    return tmp


def print_msg(msg, name=""):
    """
    Print the msg out in a formatted way
    :param msg: Msg to be printed
    :param name: Prefix name for identifying the scripts
    :return:
    """
    header = "[%s] " % name
    msg = header + msg
    print(msg)


def load_json(path):
    """
    Load the JSON Object from the path
    :param path: Path of the JSON Object
    :return: JSON Object
    """
    with open(path, errors='replace') as rfile:
        return json.load(rfile)


def load_gzip_as_json(path):
    """
    Load a GZIP file as JSON
    :param path: Path of GZIP file
    :return:
    """
    with gzip.open(path, 'r') as fin:
        json_bytes = fin.read()
    json_str = json_bytes.decode('utf-8')
    return json.loads(json_str)


def load_jsonlgz_as_json(path):
    """
    Load a JSONL GZIP file as JSON
    :param path: Path of GZIP file
    :return:
    """
    with gzip.open(path, 'r') as fin:
        json_list = list(fin)
    json_items = []
    for json_str in json_list:
        result = json.loads(json_str)
        json_items.append(result)
    return json_items


def save_json(obj, path):
    """
    Save the JSON object to the Path
    :param obj: JSON Object
    :param path: Path of the Saving File
    :return: NIL
    """
    with open(path, 'w', errors='replace') as outfile:
        json.dump(obj, outfile, ensure_ascii=False, indent=4)


def save_json_to_gzip(dict_obj, gzip_path):
    """
    Save the dictionary object to GZIP Object
    :param dict_obj: Dictionary Object
    :param gzip_path: GZIP Path
    :return: NIL
    """
    json_str = json.dumps(dict_obj)
    json_bytes = json_str.encode('utf-8')

    with gzip.open(gzip_path, 'w') as fout:
        fout.write(json_bytes)


def view_json(json_data):
    """
    View the JSON in the list/Dict
    :param json_data: JSON/List
    :return:
    """
    if type(json_data) == list:
        print("JSON Data is a list")
        for json_dict in json_data:
            _view_json(json_dict)
        return
    print("JSON Data is a dictionary")
    _view_json(json_data)


def _view_json(json_dict):
    """
    View JSON dictionary
    :param json_dict: JSON Dictionary
    :return:
    """
    for item, value in json_dict.items():
        print("%s: %s" % (item, str(value)))
    print("=" * 256)


def torch_setup(gpuid, seed=4096, num_threads=8):
    """
    Do some setup for reproductible and gpu id
    :param gpuid: GPU ID for the data
    :param seed: Seed for Random
    :param num_threads: Limit the number of Pytorch threading
    :return: NIL
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
    seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(num_threads)


def get_pretty_metric(metrics):
    """
    Pretty way to print metrics. Pretty Format the Score Dictionary
    so that we can record them nicely
    :param metrics: Metrics that are output by the Evaluation.evaluate.py
    :return: Print the Message
    """
    out_str = ""
    if metrics is None:
        return out_str
    for key, value in metrics.items():
        if type(value) == str:
            out_str += "{}: {} |".format(key, value)
        else:
            out_str += "{}: {:.4f} |".format(key, value)
    return out_str


def check_if_path_exists(path):
    """
    Check if the path exists. If not, exit the program
    :param path: Path to be checked, Checking Path
    :return: NIL
    """
    if not os.path.exists(path):
        print_msg("%s does not exists. Please ensure file does exist." % path, name="Util")
        exit(-1)
    return None


def load_vocab_dict(path):
    """
    Get Vocab Dictionary based on mode string
    :param path: Path of Vocabulary
    :return:
    """
    check_if_path_exists(path)
    if path.endswith('.bpe'):
        tmp = BPEVocabDict("", path)
    else:
        tmp = VocabDict("", path)
    tmp.load()
    return tmp


def get_statistics(len_list):
    """
    Get the statistics of the len_list
    :param len_list: A list of length of the sequences
    :return: Return avg, median, max, min and std
    """
    avg = sum(len_list) / float(len(len_list))
    med = statistics.median(len_list)
    max_ = max(len_list)
    min_ = min(len_list)
    std = statistics.stdev(len_list)
    return avg, med, max_, min_, std


def get_logger(log_path):
    """
    Create logger for logging
    :param log_path: Output of the log file
    :return: Logger
    """
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[fh, ch],
        format="[%(asctime)s][%(levelname)s]: %(message)s",
        datefmt=timestamp
    )
    return logging.getLogger(__name__)


# Project Specified Utility Function
def get_function_dict(config):
    """
    Get all function file path with their function id as the key
    :param config: Configuration file
    :return: Return a dictionary with the function ID as key and the function (.c) as value
    """
    function_dict = dict()
    for path in os.listdir(config.functions_folder):
        abs_path = os.path.join(config.functions_folder, path)
        fid = path[:-2]
        if fid not in function_dict:
            function_dict[fid] = []
        function_dict[fid].append(abs_path)
    return function_dict


def get_json_dict(config):
    """
    Get all JSON file path with their function id as the key
    :param config: Configuration file
    :return: Return a dictionary with the function ID as key and the json (.json) as value
    """
    json_dict = dict()
    for path in os.listdir(config.json_folder):
        abs_path = os.path.join(config.json_folder, path)
        fid = "_".join(path.split("_")[:2])
        if fid not in json_dict:
            json_dict[fid] = []
        json_dict[fid].append(abs_path)
    return json_dict


def simple_plot_dgl_graph(g, label=False):
    """
    Plot the G Graph Simpler, without any labels and nodes
    :param g: DGL Graph
    :param label: Print Labels
    :return:
    """
    g = g.cpu().to_networkx()
    # this plot requires pygraphviz package
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=label, node_size=10,
            node_color=[[.5, .5, .5]], arrowsize=4)
    plt.show()
    plt.savefig("./tmp/sample.png")


def is_tfidf(exp_mode):
    if exp_mode in [ModelType.XGBoost,
                    ModelType.NaiveBayes,
                    ModelType.SVM]:
        return True
    return False
