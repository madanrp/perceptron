#!/usr/bin/python3

from collections import Counter
from collections import defaultdict
import sys
import os
from random import shuffle
import operator

class AveragePerceptron:
    def __init__(self, num_iter=30):
        self.weights = defaultdict(Counter)
        self.avg_weights = defaultdict(Counter)
        self.num_iter = num_iter
        self.classes = set()
        self.features = set()

    def add_feature(self, feature):
        self.features.add(feature)

    def add_class(self, class_name):
        self.classes.add(class_name)

    def get_features(self):
        return self.features

    def get_classes(self):
        return self.classes

    def get_ft_weight(self, class_name, feature):
        return self.weights[class_name][feature]

    def get_avg_ft_weight(self, class_name, feature):
        return self.avg_weights[class_name][feature]

    def set_ft_weight(self, class_name, feature, weight):
        self.weights[class_name][feature] = weight

    def set_avg_ft_weight(self, class_name, feature, weight):
        self.avg_weights[class_name][feature] = weight

    def add_avg_ft_weight(self, class_name, feature, weight):
        self.avg_weights[class_name][feature] += weight

    def add_ft_weight(self, class_name, feature, weight):
        self.weights[class_name][feature] += weight

    def get_weight(self, class_name, features):
        weight = 0
        for feature in features:
            weight += self.get_ft_weight(class_name, feature)
        return weight

    def get_avg_weight(self, class_name, features):
        weight = 0
        for feature in features:
            weight += self.get_avg_ft_weight(class_name, feature)
        return weight

    def add_weight(self, class_name, features, weight):
        for feature in features:
            self.add_ft_weight(class_name, feature, weight)

    def get_label(self, features):
        weights = []
        for each_class in self.classes:
            weight = self.get_weight(each_class, features)
            weights.append((each_class, weight))
        
        weights = sorted(weights, key = operator.itemgetter(1), reverse=True)   
        return weights[0][0]

    def get_test_label(self, features):
        weights = []
        for each_class in self.classes:
            weight = self.get_avg_weight(each_class, features)
            weights.append((each_class, weight))
        
        weights = sorted(weights, key = operator.itemgetter(1), reverse=True)   
        return weights[0][0]
    
    def learn(self, examples):
        for example in examples:
            label, features = example[0], example[1:]
            for feature in features:
                for class_name in self.classes:
                    if feature not in self.weights[class_name]:
                        self.weights[class_name][feature] = 0.0

            predicted_label = self.get_label(features) 
            if label != predicted_label:
                self.add_weight(label, features, 1)
                self.add_weight(predicted_label, features, -1)

    def test_dev(self, dev_examples):
        numerator = 0
        denominator = len(dev_examples)
        for example in dev_examples:
            label, features = example[0], example[1:]
            predicted_label = self.get_label(features) 
            if label != predicted_label:
                numerator += 1
        return numerator * 100 / denominator

    def test(self, features):
        label = self.get_test_label(features)
        return label

    def do_average(self):
        for class_name in self.classes:
            self.avg_weights[class_name].update(self.weights[class_name]) 

    def normalize(self, denom):
        for class_name in self.classes:
            for key in self.avg_weights[class_name]:
                self.avg_weights[class_name][key] /=  denom

    def train(self, examples, dev_examples):
        error = 100
        num_iter = 0
        if len(dev_examples) == 0:
            dev_examples = examples

        while error > 0 and num_iter < self.num_iter:
            shuffle(examples)
            self.learn(examples)
            self.do_average()
            prev_error = error
            error = self.test_dev(dev_examples)
            print("num_iteration=%d"%num_iter)
            print("curr_error=%f:prev_error=%f"%(error, prev_error))
            num_iter += 1

        print("Number of iterations = %d" % num_iter) 
        self.normalize(num_iter*len(examples))
