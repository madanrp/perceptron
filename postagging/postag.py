#!/usr/bin/python3
import codecs
import sys
sys.path.append("../")

from percep import AveragePerceptron
import util
import get_args
import operator
import io

NEW_LINE = "\n"
START_WORD = "<WORD>"
START_POS = "<POS>"
START_NER = "<NER>"

END_WORD = "<ENDWORD>"
END_POS = "<ENDPOS>"

def get_features(line, perceptron):
    word_history = []
    pos_history = []
    result = []
    words = line.split(" ")
    num_words = len(words)
    result = []
    for index, word in enumerate(words):
        prev_word = "prev_w|"
        prev_tag = "prev_t|"

        prev2_word, prev2_tag, prev2_ner = "prev2_w|", "prev2_t|", "prev2_n|"
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


        word_history.append(word)

        curr_word = word
        word = "curr_w|"+word


        if index < num_words - 1:
            next_word = words[index + 1]
        else:
            next_word = END_WORD

        next_word = "next_w|" + next_word

        shape = "shape|" + util.get_word_shape(curr_word)

        suffix = "suffix|" + util.suffix(curr_word)
        suffix2 = "suffix2|" + curr_word[-2:]

        features = [prev_word, prev2_word, word, next_word, shape, suffix, suffix2]
        #print(features)

        tag = perceptron.test(features)
        pos_history.append(tag)

        result.append("%s/%s" % (curr_word, tag))


    return result 

def read_model_file(model_file, perceptron):
    lines = [line.strip() for line in codecs.open(model_file, "r")]

    class_label, class_str = lines[0].split("\t")
    classes = class_str.split(" ")
    for class_name in classes:
        perceptron.add_class(class_name)
    #print(classes)

    for line in lines[1:]:
        feature, weight_str = line.split("\t")
        weights = weight_str.split(" ")
        for index, weight in enumerate(weights):
            perceptron.set_avg_ft_weight(classes[index], feature, float(weight))
    
    #print(perceptron.avg_weights)


def classify(perceptron):
    input_stream = io.TextIOWrapper(sys.stdin.buffer)
    #for line in sys.stdin:
    for line in input_stream:
        line = line.strip()
        tags = get_features(line, perceptron)
        print(' '.join(tags))
        sys.stdout.flush()

if __name__ == "__main__":
    args = get_args.get_test_args()
    model_file = args.MODELFILE
    perceptron = AveragePerceptron()    
    read_model_file(model_file, perceptron)
    classify(perceptron)
