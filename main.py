
import numpy as np
import sys
from funcy import collecting
from functools import reduce
from collections import defaultdict
from typing import NamedTuple
import math

"""
Author: Brian Peterson
"""


"""
For the formulas used in this classifier, I used
"Speech and Language Processing", 3rd Edition draft,
by Daniel Jurafsky and James H. Martin
"""

POSITIVE = '1'
NEGATIVE = '0'
LABELED_TEST_DATA = "label_test_data.txt"
DEV_DATA = "training_files/dev_file.txt"


class LabelSet(NamedTuple):
    """ A class to represent a pair of gold 
        and classified labels """
    gold: int
    classified: int

    @classmethod
    def from_iterable(cls, iterable):
        """ Given an iterable of the form [gold, classified],
            make a LabelSet with the values interpreted 
            as correct types (integers)"""
        return cls._make(
            map(lambda str: int(str), iterable)
        )


@collecting
def generate_tuples_from_file(training_file_path):
    """Given the path to a file containing
       tab-separated data in 3 columns, returns a 
       representation of each line as a 3-tuple

    Args:
        training_file_path (string): A path to a file

    Returns [does NOT yield]:
        list[tuple]: a list of 3-tuples 
                     corresponding to the file's data
    """
    with open(training_file_path) as f:
        for line in f.readlines():
            sline = line.strip()
            info = sline.split('\t')
            if len(info) == 3:  # Ignore empty lines
                yield tuple(info)


def precision(gold_labels, classified_labels):
    """Calculate precision for these
        gold and classified labels

    Args:
        gold_labels (list[string]): A list of true "0" or "1" values 
        classified_labels (list[string]): A list of estimated "0" or "1" values

    Returns:
        float: precision, or true positives / (true positives + false positives)
    """
    tp, fp, _ = get_confusion_for_gold_and_classified(
      gold_labels, classified_labels)
    return tp / (tp + fp)


def recall(gold_labels, classified_labels):
    """Calculate recall for these
        gold and classified labels 

    Args:
        gold_labels (list[string]): A list of true "0" or "1" values 
        classified_labels (list[string]): A list of estimated "0" or "1" values

    Returns:
        float: recall, or true positives / (true positives + false negatives)
    """
    tp, _, fn = get_confusion_for_gold_and_classified(
      gold_labels, classified_labels)
    return tp / (tp + fn)


def f1(gold_labels, classified_labels):
    """Calculate f1 for these true and estimated labels

    Args:
        gold_labels (list[string]): A list of true "0" or "1" values
        classified_labels (list[string]): A list of estimated "0" or "1" values

    Returns:
        float: the f1, or 2*precision*recall / (precision + recall)
    """
    p = precision(gold_labels, classified_labels)
    r = recall(gold_labels, classified_labels)
    return (2 * p * r) / (p + r)


def get_confusion_for_gold_and_classified(gold_labels, classified_labels):
    """ Calculate the confusion matrix for gold
        and classified labels, as a 3-tuple (ignore true negatives) """
    return reduce(_reduce_confusion_for_label_set,
                  map(LabelSet.from_iterable,
                  zip(gold_labels, classified_labels)),
                  (0, 0, 0))


def _reduce_confusion_for_label_set(accumulator, label_set):
    """ Given an initial count of true positives, 
        false positives, and false negatives as a 3-tuple
        called accumulator, return a new 3-tuple with 
        these values updated given the next label_set. """
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

    """A class for doing basic Naive Bayes sentiment analysis 
       by training on some examples, recording counts of positively
       and negatively associated words, keeping a vocabulary,
       and calculating a prior probability of positive or negative 
       classification. 
       
       After training has been done, we can classify new sentences
       as either "1" or "0", positive or negative. """

    def __init__(self):
        # NOTE: Deal with not present words in opposite dict here
        #       by assuming a count of 0 for each word
        self.pos = defaultdict(lambda: 0)
        self.neg = defaultdict(lambda: 0)
        self.vocab = 0
        self.prior_prob_pos = 0
        self.prior_prob_neg = 0

    def train(self, examples):
        """Given some examples (with labels), record the count of positive
           and negatively associated words, a prior probability of 
           positive or negative classification, and record the total
           vocab.

        Args:
            examples (list[tuple]): A list of 3-tuples representing one 
                                    example each in the form, 
                                    (id: string, sentence: string, 
                                     label: string)
        """
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
        """ Calculate the log probability of this data
            being labeled as the given label. 
            Data is in the form (id: string, sentence: string).
            Perform laplace smoothing if a word is unknown
            in one of the two class contexts, and ignore it
            if it is entirely unknown. """
        data_labeled = (data[0], data[1], label)
        prior = (self.prior_prob_pos if label == POSITIVE 
                 else self.prior_prob_neg)
        counts = self.pos if label == POSITIVE else self.neg
        smoothed_count_of_all_word_occurences = \
            sum(counts.values()) + len(self.vocab)

        prob = math.log(prior)
        for f in self.featurize(data_labeled):
            # NOTE: Ignore words if they are completely unknown
            token = f[0]
            if token not in self.vocab:
                continue
            else: 
                # Perform laplace smoothing if word is unknown in 
                # up to one context, with a vocab size = 
                # unique tokens for all classes
                f_prob = ((counts[token] + 1) / 
                           smoothed_count_of_all_word_occurences)
                prob += math.log(f_prob)
        return prob

    def score(self, data):
        """Construct an example / datum such that the sentence
            is given as positive, and one as negative, and score each.
            Data is in the form (id: string, sentence: string)."""
        return {
          POSITIVE: self._score_as(data, POSITIVE),
          NEGATIVE: self._score_as(data, NEGATIVE)
        }

    def classify(self, data):
        """ Return "0" or "1" based on training data """
        # NOTE: Break ties by just choosing 0
        probs = self.score(data)
        return str(int(probs['1'] > probs['0']))

    @collecting
    def featurize(self, data):
        """ Given data, a 3-tuple, split the string contained in 
            data[1] into 2-tuples, each containing a token,
            and data[2], the original data's label. 
            Return this list (see collecting decorator). """
        # NOTE: Can improve here, e.g. lemmatize, 
        #       disclude stop words, etc. -- in Improved version only
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

