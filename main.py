
import numpy as np
import sys
from funcy import collecting
from collections import Counter, defaultdict 
import math

"""
Your name and file comment here:
"""


"""
Cite your sources here:
"""

"""
Implement your functions that are not methods of the Sentiment Analysis class here
"""

def generate_tuples_from_file(training_file_path):
    data = []
    with open(training_file_path) as f:
        for line in f.readlines():
            info = line.split('\t')
            if len(info) == 3:
                data.append(tuple(info))


def precision(gold_labels, classified_labels):
  pass


def recall(gold_labels, classified_labels):
  pass


def f1(gold_labels, classified_labels):
  pass


class SentimentAnalysis:

  def __init__(self):
    self.pos = defaultdict(lambda: 0)
    self.neg = defaultdict(lambda: 0)
    self.total_pos = 0
    self.total_neg = 0
    # Init some probabilities too...
    # Need prob for the each word
    # and the probability for each class (a prior)
    self.prior_prob_pos = 0
    self.prior_prob_neg = 0

  def train(self, examples):
      for e in examples:
          label = e[2]
          for f in self.featurize(e):
              token = f[0]
              if label == 0:
                  self.pos[token] += 1 
                  self.total_pos += 1
              else:
                  self.neg[token] += 1
                  self.total_neg += 1

      # Derive probabilities, maybe

  def score(self, data):
      # Deal with laplace smoothing, with a vocab size = unique tokens for all classes
      # Deal with not present words in opposite dict, maybe
      data_pos = (data[0], data[1], 1)
      data_neg = (data[0], data[1], 0)

      # Maybe use log probs, not clear
      # TODO: Ignore unseen words in examples when you are classifying them
      prob_pos = self.prior_prob_pos 
      for f in self.featurize(data_pos) 
          token = f[0]
          prob_w = self.pos[token]
          prob_pos *= prob_w 
      
      prob_neg = self.prior_prob_neg
      for f in self.featurize(data_neg)
          token = f[0]
          prob_w = self.neg[token]
          prob_neg *= prob_w

      return (prob_pos, prob_neg)

  def classify(self, data):
      # Break ties by just choosing 0
      pos_prob, neg_prob = self.score(data)
      if pos_prob > neg_prob:
          return 1
      else:
          return 0

  @collecting
  def featurize(self, data):
      # Can improve here, e.g. lemmatize
      tokens = data[1].split()
      level = data[2]
      for token in tokens:
          yield (token, level)

  def __str__(self):
    return "Naive Bayes - bag-of-words baseline"


class SentimentAnalysisImproved:

  def __init__(self):
    pass

  def train(self, examples):
    pass

  def score(self, data):
    pass

  def classify(self, data):
    pass

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
  sa.train(training)
  # Classify each example in the given testing file
  # Put the results in label_test_data.txt
  # Report precision, recall, and f1 on the test data for each of your model(s)

  improved = SentimentAnalysisImproved()
  print(improved)
  sa.train(training)

