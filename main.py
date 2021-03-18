
import numpy as np
import sys
from funcy import collecting, post_processing
from functools import reduce
from itertools import groupby
from collections import Counter
from typing import NamedTuple
import math
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

"""
Author: Brian Peterson
See the class comment of SentimentAnalysis
for description of this module's purpose.
"""


"""
For the formulas used in this classifier, I used
"Speech and Language Processing", 3rd Edition draft,
by Daniel Jurafsky and James H. Martin
"""

POSITIVE = '1'
NEGATIVE = '0'
NEVER_FEATURE = "never"
WORST_FEATURE = "worst"
DESK_FEATURE = "desk" # This is very domain-specific, but good for this domain
GREAT_FEATURE = "great"
BEST_FEATURE = "best"
BEAUTIFUL_FEATURE = "beautiful"
WONDERFUL_FEATURE = "wonderful"
FABULOUS_FEATURE = "fabulous"
FANTASTIC_FEATURE = "fantastic"
LOVE_FEATURE = "love"
FRIENDLY_FEATURE = "friendly"
PERFECT_FEATURE = "perfect" 
RELAX_FEATURE = "relax"
HELPFUL_FEATURE = "helpful"
SUBPAR_FEATURE = "sub-par"
RUDE_FEATURE = "rude"
SNOBBISH_FEATURE = "snobbish" 
EXAGGERATED_FEATURE = "exaggerated"
SAFETY_FEATURE = "safety"
FIRST_AND_SECOND_PRONOUNS = (
    "i", "me", "my", "mine", "myself", "we", 
    "us", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves"
)

LABEL_TEST_DATA = "label_test_data.txt"
IMPROVED_LABEL_TEST_DATA = "improved_label_test_data.txt"
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
            if len(info) > 1:  # Ignore empty lines
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
    tp, fp, _, _ = get_confusion_for_gold_and_classified(
      gold_labels, classified_labels)
    try:
        return tp / (tp + fp)
    except:
        return math.inf


def recall(gold_labels, classified_labels):
    """Calculate recall for these
        gold and classified labels 

    Args:
        gold_labels (list[string]): A list of true "0" or "1" values 
        classified_labels (list[string]): A list of estimated "0" or "1" values

    Returns:
        float: recall, or true positives / (true positives + false negatives)
    """
    tp, _, fn, _ = get_confusion_for_gold_and_classified(
      gold_labels, classified_labels)
    try:
        return tp / (tp + fn)
    except:
        return math.inf


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


def accuracy(gold_labels, classified_labels):
    tp, fp, fn, tn = get_confusion_for_gold_and_classified(
      gold_labels, classified_labels)
    try:
        return (tp + tn) / (tp + fp + tn + fn)
    except:
        return math.inf

def get_confusion_for_gold_and_classified(gold_labels, classified_labels):
    """ Calculate the confusion matrix for gold
        and classified labels, as a 4-tuple """
    return reduce(_reduce_confusion_for_label_set,
                  map(LabelSet.from_iterable,
                  zip(gold_labels, classified_labels)),
                  (0, 0, 0, 0))


def _reduce_confusion_for_label_set(accumulator, label_set):
    """ Given an initial count of true positives, 
        false positives, false negatives, and true negatives 
        as a 4-tuple called accumulator, return a new 4-tuple 
        with these values updated given the next label_set. """
    tp, fp, fn, tn = accumulator
    if label_set.gold and label_set.classified:
        return (tp+1, fp, fn, tn)
    elif not label_set.gold and label_set.classified:
        return (tp, fp+1, fn, tn)
    elif label_set.gold and not label_set.classified:
        return (tp, fp, fn+1, tn)
    else:
        return (tp, fp, fn, tn+1)


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


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
        self.pos = Counter()
        self.neg = Counter()
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
                try:
                    counts[token] # Check if the word is present
                                  # in this class context
                except KeyError:
                    counts[token] = 0 # If not, set its count to 0
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
        tokens = data[1].split()
        label = data[2]
        for token in tokens:
            yield (token, label)

    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"


class SentimentAnalysisImproved(SentimentAnalysis):

                #      pos   neg  never desk worst   safety sub-par rudesnob exagg great best beautiful wonderful fabfan  friendly love perfect relax 1st2ndPN log#words   bias
    INITIAL_WEIGHTS = [3,   -3,  -1.5, -1.5,  -2,     -1,   -1,      -2,      -1,   3,   3,     3,        3,       4,      2,       2,    2,       2,    -0.2,     0.2,      1]

    def __init__(self):
        super().__init__()
        # Some initial weights
        self.weights = np.array(self.INITIAL_WEIGHTS)
        self.learning_rate = 0.1 # This is relatively low: only move 
                                 # initial weights much if they are pretty far off
        # The batch size will be 10% of all examples (= 10 iterations)
        self.relative_batch_size = 0.10
        self.threshold = 0.5

    def train(self, examples):
        super().train(examples)

        # TODO: use bigram counts as well

        # Find most common words
        self.most_common_number = 200
        self.most_common_pos = dict(self.pos.most_common(self.most_common_number))
        self.most_common_neg = dict(self.neg.most_common(self.most_common_number))

        # Perform gradient descent to find best weights
        # Form batches of batch_size
        # NOTE: they could actually be mostly batch_size-1 
        #       because of how using modulo to make groups works
        batch_size = math.floor(len(examples) * self.relative_batch_size)
        indexed_examples = \
                [(i, example) for i, example in enumerate(examples)]
        for key, example_group in groupby(
                indexed_examples, 
                key=lambda x: x[0] % batch_size):
            
            losses = np.array([0.0] * len(self.weights))
            for indexed_example in example_group:
                example = indexed_example[1] # Drop the wrapper index tuple
                losses += self.get_cross_entropy_loss_gradient(example)
            self.update_weights(losses)
        # import pdb; pdb.set_trace()

    @collecting
    def featurize(self, data):
        """ Given data, a 3-tuple, split the string contained in 
            data[1] into 2-tuples, each containing a token,
            and data[2], the original data's label. 
            Return this list (see collecting decorator). """
        tokens = word_tokenize(data[1])
        lemmatizer = WordNetLemmatizer()
        label = data[2]
        for token in tokens:
            token = lemmatizer.lemmatize(token)
            token = token.lower()
            yield (token, label)
    
    def update_weights(self, loss_gradient):
        # Implement gradient descent
        self.weights -= (
            self.learning_rate * 
            loss_gradient
        )

    @post_processing(np.array)
    @collecting
    def get_cross_entropy_loss_gradient(self, example):
        label = example[2]
        features = self.get_data_features(example)
        class_prob = sigmoid(features.dot(self.weights))
        error = class_prob - int(label)
        for feature in features:
            yield error * feature

    def get_data_features(self, data):
        data = data[1]

        tokens = word_tokenize(data)
        tokens_count = Counter(tokens)
        percent_pos = len([token for token in tokens
                                  if token in self.most_common_pos]) \
                         / self.most_common_number
        percent_neg = len([token for token in tokens
                                  if token in self.most_common_neg]) \
                         / self.most_common_number
        never_feature = int(NEVER_FEATURE in tokens)
        subpar_feature = int(SUBPAR_FEATURE in tokens)
        safety_feature = int(SAFETY_FEATURE in tokens) 
        desk_feature = int(DESK_FEATURE in tokens)
        worst_feature = int(WORST_FEATURE in tokens)
        friendly_feature = int(FRIENDLY_FEATURE in tokens) + int(HELPFUL_FEATURE in tokens)
        great_feature = int(GREAT_FEATURE in tokens)
        best_feature = int(BEST_FEATURE in tokens)
        beautiful_feature = int(BEAUTIFUL_FEATURE in tokens)
        wonderful_feature = int(WONDERFUL_FEATURE in tokens)
        love_feature = int(LOVE_FEATURE in tokens)
        fabulous_fantastic_feature = int(FABULOUS_FEATURE in tokens) +      \
                                         int(FANTASTIC_FEATURE in tokens)
        perfect_feature = int(PERFECT_FEATURE in tokens)
        relax_feature = int(RELAX_FEATURE in tokens)
        rude_snobbish_feature = int(RUDE_FEATURE in tokens) + int(SNOBBISH_FEATURE in tokens)
        exaggerated_feature = int(EXAGGERATED_FEATURE in tokens)
        num_first_second_pronouns = len([token for token in tokens
                                         if token in FIRST_AND_SECOND_PRONOUNS])
        log_word_count_of_doc = math.log(len(tokens))
        bias = self.prior_prob_pos - self.prior_prob_neg
        return np.array([
            percent_pos, percent_neg, never_feature, desk_feature, 
            worst_feature, safety_feature, subpar_feature, 
            rude_snobbish_feature, exaggerated_feature,
            great_feature, best_feature, beautiful_feature, 
            wonderful_feature, fabulous_fantastic_feature,
            friendly_feature, love_feature, perfect_feature, relax_feature,
            num_first_second_pronouns, log_word_count_of_doc, bias
        ]) 

    def classify(self, data):
        return str(int(self.score(data) > self.threshold))

    def score(self, data):
        features = self.get_data_features(data) 
        return sigmoid(features.dot(self.weights))

    def __str__(self):
        return "Logistic Regression"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
        sys.exit(1)

    training = sys.argv[1]
    testing = sys.argv[2]

    sa = SentimentAnalysis()
    print(sa)
    sa.train(generate_tuples_from_file(training))

    # Classify each example in the given testing file
    # using the basic model
    # Put the results in label_test_data.txt
    with open(LABEL_TEST_DATA, 'w') as output:
        for example in generate_tuples_from_file(testing):
            label = sa.classify(example)
            output.write(f'{example[0]} {label}\n')

    # Report precision, recall, and f1 on the test data 
    # (dev_file) for basic model
    gold_labels = []
    classified_labels = []
    labels = (gold_labels, classified_labels)
    for example in generate_tuples_from_file(DEV_DATA):
        gold_labels.append(example[2])
        classified_labels.append(sa.classify(example))

    print(gold_labels)
    print(classified_labels)
    print(f'recall: {recall(*labels)}')
    print(f'precision: {precision(*labels)}')
    print(f'f1: {f1(*labels)}')
    print(f'accuracy: {accuracy(*labels)}')

    improved = SentimentAnalysisImproved()
    print(improved)
    improved.train(generate_tuples_from_file(training))

    # Classify each example in the given testing file using 
    # the improved models
    # Put the results in label_test_data.txt
    with open(IMPROVED_LABEL_TEST_DATA, 'w') as output:
        for example in generate_tuples_from_file(testing):
            label = sa.classify(example)
            output.write(f'{example[0]} {label}\n')

    # Report precision, recall, and f1 on the test data 
    # (dev_file) for improved model

    gold_labels = []
    classified_labels = []
    labels = (gold_labels, classified_labels)
    for example in generate_tuples_from_file(DEV_DATA):
        gold_labels.append(example[2])
        classified_labels.append(improved.classify(example))

    print(gold_labels)
    print(classified_labels)
    print(f'recall: {recall(*labels)}')
    print(f'precision: {precision(*labels)}')
    print(f'f1: {f1(*labels)}')
    print(f'accuracy: {accuracy(*labels)}')



