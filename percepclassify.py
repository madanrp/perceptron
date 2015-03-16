#!/usr/bin/python3
import sys
import operator
from perceptron import Perceptron
from optparse import OptionParser
import get_args

LAST_WORD = "<LAST>"
START_TAG = "<START>"
NEW_LINE = "\n"
TAB_SPACE = "\t"

def read_model_file(model_file, perceptron):
    with open(model_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        if len(lines) == 0:
            return
        classes = lines[0].split(TAB_SPACE)[1]
        classes = classes.split()
        for class_name in classes:
            perceptron.add_class(class_name)

        perceptron.set_weights()

        features = lines[1:]
        for feature in features:
            feature, weight_str = feature.split(TAB_SPACE)
            perceptron.add_feature(feature)
            weights = weight_str.split()
            for index, weight in enumerate(weights):
                class_name = classes[index] 
                perceptron.set_avg_feature_weight(class_name, feature, float(weight))

def get_tag(sentence, perceptron):
    words = sentence.split()
    predicted_tag = perceptron.test(words)
    return predicted_tag

def classify(perceptron):
    for line in sys.stdin:
        tag = get_tag(line, perceptron)
        print(tag)
        sys.stdout.flush()

if __name__ == "__main__":

    #if len(sys.argv) < 2 or len(sys.argv) > 2:
    #    print("Usage: python3 postag.py MODELFILE")
    #    sys.exit(-1)

    #model_file = sys.argv[1]

    args = get_args.get_test_args()
    model_file = args.MODELFILE
    perceptron = Perceptron()    
    read_model_file(model_file, perceptron)
    classify(perceptron)
