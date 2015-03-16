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
    lines = [line.strip() for line in codecs.open(file_name, "r", encoding="latin1").readlines()]
    examples = []
    feature_set = set()
    classes = set()
    for line in lines:
        word_history = []
        ner_history = []
        pos_history = []
        words = line.split(" ")
        num_words = len(words)
        for index, word in enumerate(words):
            prev_word = "prev_w|"
            prev_tag = "prev_t|"
            prev_ner = "prev_n|"

            prev2_word, prev2_tag, prev2_ner = "prev2_w|", "prev2_t|", "prev2_n|"
            if index - 1 < 0:
                prev_word += START_WORD
                prev_tag += START_POS
                prev_ner += START_NER
            else:
                prev_word += word_history[index - 1]
                prev_tag += pos_history[index - 1]
                prev_ner += ner_history[index - 1]

            if index - 2 < 0:
                prev2_word += START_WORD
                prev2_tag += START_POS
                prev2_ner += START_NER
            else:
                prev2_word += word_history[index - 2]
                prev2_tag += pos_history[index - 2]
                prev2_ner += ner_history[index - 2]

            word, tag, ner = word.rsplit("/", 2)

            word_history.append(word)
            ner_history.append(ner)
            pos_history.append(tag)

            curr_word  = word
            word, tag = "curr_w|"+word, "curr_t|"+tag


            if index < num_words - 1:
                next_word, next_tag, next_ner = words[index + 1].rsplit("/", 2)
            else:
                next_word, next_tag, next_ner = END_WORD, END_POS, "END"

            next_word = "next_w|" + next_word
            next_tag = "next_t|" + next_tag

            shape = "shape|" + util.get_word_shape(curr_word)
            suffix = "suffix|" + util.suffix(curr_word)

            features = [ner, prev_word, prev2_word, word, next_word, tag, prev_tag, prev2_tag, next_tag, prev_ner, prev2_ner, shape, suffix]
            #print(features)
            examples.append(features)
            for feature in features:
                feature_set.add(feature)

            classes.add(ner)


    return examples, classes, feature_set

def write_learnt_data(model_file, perceptron):
    with codecs.open(model_file, "w", encoding="latin1") as f:
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
