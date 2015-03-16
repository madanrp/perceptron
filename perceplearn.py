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

def read_training_file(file_name, perceptron):
    lines = [ line.strip() for line in open(file_name).readlines()]
    training_set = []
    for line in lines:
        words = line.split()
        tag = words[0]
        training_set.append(words)

        if not perceptron.check_class(tag):
            perceptron.add_class(tag)

        for feature in words[1:]:
            if not perceptron.check_feature(feature): 
                perceptron.add_feature(feature)
    
    return training_set

def write_learnt_data(model_file, perceptron):
    with open(model_file, "w") as f:
        classes = sorted(perceptron.get_classes())
        f.write("CLASSES" + "\t" + ' '.join(classes))
        f.write(NEW_LINE)

        features = perceptron.get_features()
        sorted_features = sorted(features.items(), key = operator.itemgetter(1))
        
        for feature, index in sorted_features:
            f.write(feature)
            f.write(TAB_SPACE)
            weights = []
            for class_name in classes:
                weights.append(str(perceptron.get_avg_feature_weight(class_name, feature)))
            f.write(' '.join(weights))
            f.write(NEW_LINE)

if __name__ == "__main__":
    #args = sys.argv
    ##if len(args) != 3:
    ##    print('perceplearn.py TRANINGFILE MODELFILE')
    ##    sys.exit(2)

    #usage = "usage: %prog [options] TRAININGFILE MODELFILE"
    #parser = OptionParser(usage=usage)
    #parser.add_option("-d", "--dev",
    #                    help="dev file", metavar="FILE") 

    #(options, args) = parser.parse_args()

    #if  len(args) != 2:
    #    print("perceplearn.py TRANINGFILE MODELFILE")
    #    sys.exit(2)

    #dev_file = None
    #if options.dev is not None:
    #    dev_file = options.dev

    #training_file = args[0]
    #model_file = args[1]
    args = get_args.get_train_args()

    dev_file = args.DEVFILE
    training_file = args.TRAININGFILE
    model_file = args.MODELFILE

    perceptron = Perceptron()
    perceptron.set_num_iterations(30)
    training_set = read_training_file(training_file, perceptron)
    dev_set = []
    if dev_file is not None:
        dev_set = read_training_file(dev_file, perceptron)
    perceptron.set_weights()
    perceptron.train(training_set, dev_set)
    write_learnt_data(model_file, perceptron)
