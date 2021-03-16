
import numpy as np
import sys
from funcy import collecting
from functools import reduce
from collections import Counter, defaultdict
from typing import NamedTuple
import math

"""
Your name and file comment here:
"""


"""
Cite your sources here:
"""

POSITIVE = '1'
NEGATIVE = '0'
LABELED_TEST_DATA = "label_test_data.txt"
DEV_DATA = "training_files/dev_file.txt"

class LabelSet(NamedTuple):
    gold: int
    classified: int

    @classmethod
    def from_iterable(cls, iterable):
      return cls._make(
        map(lambda str: int(str), iterable)
      )


@collecting
def generate_tuples_from_file(training_file_path):
    with open(training_file_path) as f:
        for line in f.readlines():
            sline = line.strip()
            info = sline.split('\t')
            if len(info) == 3:  # Ignore empty lines
                yield tuple(info)


def precision(gold_labels, classified_labels):
    tp, fp, _ = get_confusion_for_gold_and_classified(
      gold_labels, classified_labels)
    return tp / (tp + fp)


def recall(gold_labels, classified_labels):
    tp, _, fn = get_confusion_for_gold_and_classified(
      gold_labels, classified_labels)
    return tp / (tp + fn)


def f1(gold_labels, classified_labels):
    p = precision(gold_labels, classified_labels)
    r = recall(gold_labels, classified_labels)
    return (2 * p * r) / (p + r)


def get_confusion_for_gold_and_classified(gold_labels, classified_labels):
    return reduce(_reduce_confusion_for_label_set,
                  map(LabelSet.from_iterable,
                  zip(gold_labels, classified_labels)),
                  (0, 0, 0))


def _reduce_confusion_for_label_set(accumulator, label_set):
    tp, fp, fn = accumulator
    if label_set.gold and label_set.classified:
        return (tp+1, fp, fn)
    elif not label_set.gold and label_set.classified:
        return (tp, fp+1, fn)
    elif label_set.gold and not label_set.classified:
        return (tp, fp, fn+1)
    else:
        # NOTE: We don't need to know the # of true negatives
        # to calculate precision, recall, and F1
        return (tp, fp, fn)


class SentimentAnalysis:

    def __init__(self):
        # NOTE: Deal with not present words in opposite dict here
        self.pos = defaultdict(lambda: 0)
        self.neg = defaultdict(lambda: 0)
        self.vocab = 0
        self.prior_prob_pos = 0
        self.prior_prob_neg = 0

    def train(self, examples):
        total_pos = 0
        total_neg = 0
        for e in examples:
            label = e[2]

            if label == POSITIVE:
                total_pos += 1
            else:
                total_neg += 1

            for f in self.featurize(e):
                token = f[0]
                if label == POSITIVE:
                    self.pos[token] += 1
                else:
                    self.neg[token] += 1

        # Derive probabilities
        total_count = total_pos + total_neg
        self.prior_prob_neg = total_neg / total_count
        self.prior_prob_pos = total_pos / total_count
        self.vocab = set(self.neg.keys()) | set(self.pos.keys())

    def _score_as(self, data, label):
        data_labeled = (data[0], data[1], label)
        prior = (self.prior_prob_pos if label == POSITIVE 
                 else self.prior_prob_neg)
        counts = self.pos if label == POSITIVE else self.neg
        denominator = sum(counts.values()) + len(self.vocab)

        prob = math.log(prior)
        for f in self.featurize(data_labeled):
            # NOTE: Ignore words if they are completely unknown
            token = f[0]
            if token not in self.vocab:
                continue
            else: 
                # Perform laplace smoothing if only unknown in one context, with a 
                # vocab size = unique tokens for all classes
                f_prob = (counts[token] + 1) / denominator
                prob += math.log(f_prob)
        return prob

    def score(self, data):
        """ Stuff """
        # Construct an example / datum such that the sentence
        # is given as positive, and one as negative, and score each
        return {
          POSITIVE: self._score_as(data, POSITIVE),
          NEGATIVE: self._score_as(data, NEGATIVE)
        }

    def classify(self, data):
        # NOTE: Break ties by just choosing 0
        probs = self.score(data)
        return str(int(probs['1'] > probs['0']))

    @collecting
    def featurize(self, data):
        # Can improve here, e.g. lemmatize, disclude stop words, etc.
        tokens = data[1].split()
        label = data[2]
        for token in tokens:
            yield (token, label)

    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"


class SentimentAnalysisImproved(SentimentAnalysis):

    def featurize(self, data):
        pass

    def __str__(self):
        return "NAME FOR YOUR CLASSIFIER HERE"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
        sys.exit(1)

    training = sys.argv[1]
    testing = sys.argv[2]

    sa = SentimentAnalysis()
    print(sa)

    # Classify each example in the given testing file
    # Put the results in label_test_data.txt
    sa.train(generate_tuples_from_file(training))
    with open(LABELED_TEST_DATA, 'w') as output:
        for example in generate_tuples_from_file(testing):
            label = sa.classify(example)
            output.write(f'{example[0]} {label}\n')

    # Report precision, recall, and f1 on the test data 
    # (dev_file) for each of your model(s)
    gold_labels = []
    classified_labels = []
    labels = (gold_labels, classified_labels)
    for example in generate_tuples_from_file(DEV_DATA):
        gold_labels.append(example[2])
        classified_labels.append(sa.classify(example))

    print(f'recall: {recall(*labels)}')
    print(f'precision: {precision(*labels)}')
    print(f'f1: {f1(*labels)}')

    #improved = SentimentAnalysisImproved()
    #print(improved)
    #improved.train(generate_tuples_from_file(training))

