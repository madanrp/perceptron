#!/usr/bin/python3
import codecs
import sys
sys.path.append("../")

from percep import AveragePerceptron
import util
import get_args
import operator

NEW_LINE = "\n"
START_WORD = "<WORD>"
START_POS = "<POS>"
START_NER = "<NER>"

END_WORD = "<ENDWORD>"
END_POS = "<ENDPOS>"

def get_features(file_name):
    lines = [line.strip() for line in codecs.open(file_name, "r").readlines()]
    examples = []
    feature_set = set()
    classes = set()
    for line in lines:
        word_history = []
        pos_history = []
        words = line.split(" ")
        num_words = len(words)
        for index, word in enumerate(words):
            prev_word = "prev_w|"
            prev_tag = "prev_t|"

            prev2_word, prev2_tag = "prev2_w|", "prev2_t|"
            if index - 1 < 0:
                prev_word += START_WORD
                prev_tag += START_POS
            else:
                prev_word += word_history[index - 1]
                prev_tag += pos_history[index - 1]

            if index - 2 < 0:
                prev2_word += START_WORD
                prev2_tag += START_POS
            else:
                prev2_word += word_history[index - 2]
                prev2_tag += pos_history[index - 2]

            word, tag= word.rsplit("/", 1)

            word_history.append(word)
            pos_history.append(tag)

            curr_word  = word
            word = "curr_w|"+word


            if index < num_words - 1:
                next_word, next_tag= words[index + 1].rsplit("/", 1)
            else:
                next_word, next_tag = END_WORD, END_POS

            next_word = "next_w|" + next_word
            next_tag = "next_t|" + next_tag

            shape = "shape|" + util.get_word_shape(curr_word)
            suffix = "suffix|" + util.suffix(curr_word)
            suffix2 = "suffix2|" + curr_word[-2:]

            features = [tag, prev_word, prev2_word, word, next_word, shape, suffix, suffix2]
            #print(features)
            examples.append(features)
            for feature in features:
                feature_set.add(feature)

            classes.add(tag)


    return examples, classes, feature_set

def write_learnt_data(model_file, perceptron):
    with codecs.open(model_file, "w") as f:
        classes = perceptron.get_classes()
        classes = sorted(classes)
        class_str = "CLASSES" + "\t" + ' '.join(classes)
        f.write(class_str)
        f.write(NEW_LINE)

        for feature in perceptron.get_features():
            weight_str = feature + "\t"
            weights = []
            for class_name in classes:
                weight = str(perceptron.get_avg_ft_weight(class_name, feature))
                weights.append(weight)
            weight_str = feature + "\t" + ' '.join(weights)
            f.write(weight_str)
            f.write(NEW_LINE)

if __name__ == "__main__":
    args = get_args.get_train_args()
    training_file = args.TRAININGFILE
    model_file = args.MODELFILE
    dev_file = args.DEVFILE
    
    perceptron = AveragePerceptron(20)
    training_set, classes, features = get_features(training_file)
    print(classes)
    #print(features)
    for class_name in classes:
        perceptron.add_class(class_name)

    for feature in features:
        perceptron.add_feature(feature)

    dev_set = []
    if dev_file is not None:
        dev_set, classes, features = get_features(dev_file)
    print(len(dev_set))
    perceptron.train(training_set, dev_set)
    write_learnt_data(model_file, perceptron)    
    #print(training_set)
