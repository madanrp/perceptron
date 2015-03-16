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
    ner_history = []
    pos_history = []
    result = []
    words = line.split(" ")
    num_words = len(words)
    result = []
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

        word, tag = word.rsplit("/", 1)

        word_history.append(word)
        pos_history.append(tag)

        curr_word, curr_tag = word, tag
        word, tag = "curr_w|"+word, "curr_t|"+tag


        if index < num_words - 1:
            next_word, next_tag= words[index + 1].rsplit("/", 1)
        else:
            next_word, next_tag = END_WORD, END_POS

        next_word = "next_w|" + next_word
        next_tag = "next_t|" + next_tag

        shape = "shape|" + util.get_word_shape(curr_word)

        suffix = "suffix|" + util.suffix(curr_word)

        features = [prev_word, prev2_word, word, next_word, tag, prev_tag, prev2_tag, next_tag, prev_ner, prev2_ner, shape, suffix]
        #print(features)

        ner = perceptron.test(features)
        ner_history.append(ner)

        result.append("%s/%s/%s" % (curr_word, curr_tag, ner))


    return result 

def read_model_file(model_file, perceptron):
    lines = [line.strip() for line in codecs.open(model_file, "r", encoding="latin1")]

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
    input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='latin1')
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
