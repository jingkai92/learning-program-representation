import argparse
from nltk.translate import meteor_score
from evaluation.eval import QGEvalCap


def evaluate_predictions(target_src, decoded_text):
    assert len(target_src) == len(decoded_text)
    eval_targets = {}
    eval_predictions = {}
    for idx in range(len(target_src)):
        eval_targets[idx] = [target_src[idx]]
        eval_predictions[idx] = [decoded_text[idx]]

    QGEval = QGEvalCap(eval_targets, eval_predictions)
    scores = QGEval.evaluate(include_meteor=False)
    # compute_meteor_score(target_src, decoded_text)
    # repair_accuracy, repair_count = compute_repair_accuracy(target_src, decoded_text)
    # scores['RAcc'] = repair_accuracy
    # scores['RCount'] = repair_count
    return scores


def compute_meteor_score(gold, pred):
    sample_num = len(pred) if len(pred) < len(gold) else len(gold)
    scores = 0
    for i in range(sample_num):
        scores += round(meteor_score.meteor_score([gold[i]], pred[i]), 4)
    average_score = scores / sample_num
    print('Meteor: %f' % average_score)


def main(args):
    pred = []
    with open(args['pred_in'], 'r', encoding='utf-8') as f1:
        for line in f1:
            pred.append(line.strip())

    gold = []
    with open(args['gold_in'], 'r', encoding='utf-8') as f2:
        for line in f2:
            gold.append(line.strip())
    metrics = evaluate_predictions(gold, pred)
    return metrics


def compute_repair_accuracy(pred, gold):
    repair_accs = []
    for i in range(len(gold)):
        pout = pred[i]
        ptgt = gold[i]
        if pout == ptgt:
            repair_accs.append(1)
        else:
            repair_accs.append(0)
    repair_count = "%i/%i" % (sum(repair_accs), len(repair_accs))
    return sum(repair_accs) / float(len(repair_accs)), repair_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gold_in', '--gold_in', required=True, type=str, help='path to the gold input file')
    parser.add_argument('-pred_in', '--pred_in', required=True, type=str, help='path to the pred input file')
    args = vars(parser.parse_args())
    main(args)

