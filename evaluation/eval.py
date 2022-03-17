#!/usr/bin/env python
__author__ = 'xinya'

# Modified by:
# Yu Chen <cheny39@rpi.edu>

# Changelog:
# Convert to python 3
# Support verbose mode
# Change output format
# Add word mover's distance
from .bleu.bleu import Bleu
from evaluation.meteor.meteor import Meteor
from evaluation.rouge.rouge import Rouge
from collections import defaultdict
# from eval import QGEvalCap
from json import encoder

# reload(sys)
# sys.setdefaultencoding('utf-8')


class QGEvalCap:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self, include_meteor=False, verbose=False):
        output = {}
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
        ]
        if include_meteor:
            scorers.append((Meteor(),"METEOR"))
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    if verbose:
                        print("%s: %0.5f"%(m, sc))
                    # output.append(sc)
                    output[m] = sc
            else:
                if verbose:
                    print("%s: %0.5f"%(method, score))
                # output.append(score)
                output[method] = score
        return output


def eval(out_file, src_file, tgt_file):
    """
        Given a filename, calculate the metric scores for that prediction file

        isDin: boolean value to check whether input file is DirectIn.txt
    """
    pairs = []
    with open(src_file, 'r') as infile:
        for line in infile:
            pair = {}
            pair['tokenized_sentence'] = line[:-1]
            pairs.append(pair)

    with open(tgt_file, "r") as infile:
        cnt = 0
        for line in infile:
            pairs[cnt]['tokenized_question'] = line[:-1]
            cnt += 1

    output = []
    with open(out_file, 'r') as infile:
        for line in infile:
            line = line[:-1]
            output.append(line)
    for idx, pair in enumerate(pairs):
        pair['prediction'] = output[idx]

    ## eval
    encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])
    for pair in pairs[:]:
        key = pair['tokenized_sentence']
        res[key] = [pair['prediction'].encode('utf-8')]
        gts[key].append(pair['tokenized_question'].encode('utf-8'))

    QGEval = QGEvalCap(gts, res)
    return QGEval.evaluate()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-out", "--out_file", dest="out_file", default="./output/pred.txt", help="output file to compare")
    parser.add_argument("-src", "--src_file", dest="src_file", default="../data/processed/src-test.txt", help="src file")
    parser.add_argument("-tgt", "--tgt_file", dest="tgt_file", default="../data/processed/tgt-test.txt", help="target file")
    args = parser.parse_args()

    print("scores: \n")
    eval(args.out_file, args.src_file, args.tgt_file)


