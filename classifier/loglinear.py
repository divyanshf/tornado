import numpy as np
import math
import operator
import random
from collections import OrderedDict
import copy
from classifier.classifier import SuperClassifier
from data_structures.attribute import Attribute
from dictionary.tornado_dictionary import TornadoDic


class LogLinear(SuperClassifier):
    """This is the implementation of a LogLinear Model for learning from data streams."""

    LEARNER_NAME = TornadoDic.LOG_LIN
    LEARNER_TYPE = TornadoDic.TRAINABLE
    LEARNER_CATEGORY = TornadoDic.NUM_CLASSIFIER

    __BIAS_ATTRIBUTE = Attribute()
    __BIAS_ATTRIBUTE.set_name("bias")
    __BIAS_ATTRIBUTE.set_type(TornadoDic.NUMERIC_ATTRIBUTE)
    __BIAS_ATTRIBUTE.set_possible_values(1)

    def __init__(self, labels, attributes, eta=1):
        attrs = copy.copy(attributes)
        attrs.append(self.__BIAS_ATTRIBUTE)

        super().__init__(labels, attrs)

        self.WEIGHTS = OrderedDict()
        self.__initialize_weights()
        self.ETA = eta

    def __initialize_weights(self):
        for c in self.CLASSES:
            self.WEIGHTS[c] = np.zeros(len(self.ATTRIBUTES))

    def CalcScores(self, x):
        max_score = 0.0
        scores = OrderedDict()
        for c in self.CLASSES:
            w = self.WEIGHTS[c]
            sc = np.sum(np.dot(w, x))
            max_score = max(max_score, sc)
            scores[c] = sc

        sum_score = 0.0
        for c in self.CLASSES:
            scores[c] = np.exp(scores[c] - max_score)
            sum_score += scores[c]

        for c in self.CLASSES:
            scores[c] /= sum_score

        return scores

    def train(self, instance):
        x = instance[0 : len(instance) - 1]
        x.append(1)
        y_real = instance[len(instance) - 1]
        scores = self.CalcScores(x)
        self.Update(x, y_real, scores)
        self._IS_READY = True

    def test(self, instance):
        if self._IS_READY:
            x = instance[0 : len(instance) - 1]
            y = instance[len(instance) - 1]
            x.append(1)
            scores = self.CalcScores(x)
            y_predicted = self.CLASSES[0]
            for k in scores:
                if scores[y_predicted] < scores[k]:
                    y_predicted = k
            self.update_confusion_matrix(y, y_predicted)
            return y_predicted
        else:
            print("Please train a LogLinear classifier first!")
            exit()

    def Update(self, x, y, scores):
        correct_weights = self.WEIGHTS[y]
        for a in range(len(self.ATTRIBUTES)):
            for c in self.CLASSES:
                w = self.WEIGHTS[c]
                w[a] -= self.ETA * scores[c] * x[a]
            correct_weights[a] += self.ETA * x[a]
        self.WEIGHTS[y] = correct_weights

    def reset(self):
        super()._reset_stats()
        self.WEIGHTS = OrderedDict()
        self.__initialize_weights()
