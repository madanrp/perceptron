#!/usr/bin/python3
import sys

def get_bio(this_tuple):
    return this_tuple[2]

def get_ner(this_tuple):
    return this_tuple[3]

def get_entities(tuples):
    num_tuple = len(tuples)
    entities = set()
    index = 0
    while index < num_tuple:
        entity_start = index
        curr_tuple = tuples[index]
        bio = get_bio(curr_tuple)
        ner = get_ner(curr_tuple)
        index += 1
        if bio != 'B':
            continue
        while index < num_tuple and get_bio(tuples[index]) == 'I' and get_ner(tuples[index]) == ner:
            index += 1

        entity_end = index -1
        entities.add((entity_start, entity_end, ner))

    return entities

expected_file = sys.argv[1]
output_file = sys.argv[2]
lines = [ line.strip() for line in open(expected_file, encoding = "latin1").readlines()]
output_lines = [ line.strip() for line in open(output_file, encoding = "ISO-8859-1").readlines()]


correct_ner_dict = {}
total_ner_dict = {}
total_output_ner_dict = {}

for line_index, line in enumerate(lines):
    words = line.split()
    output_words = output_lines[line_index].split()
    word_tuples = []
    output_word_tuples= []
    for word_index, each_word in enumerate(words):
        word, pos_tag, ner_tag = each_word.rsplit("/", 2)
        if ner_tag != 'O':
            bio, ner = ner_tag.split("-", 1)
        else:
            bio, ner = 'O', 'O'
        word_tuples.append((word, pos_tag, bio, ner))

        word, pos_tag, ner_tag = output_words[word_index].rsplit("/", 2)
        if ner_tag != 'O':
            bio, ner = ner_tag.split("-", 1)
        else:
            bio, ner = 'O', 'O'
        output_word_tuples.append((word, pos_tag, bio, ner))

    input_entities = get_entities(word_tuples)
    output_entities = get_entities(output_word_tuples)
    correct_entities = input_entities & output_entities

    for entity in input_entities:
        ner = entity[2]
        total_ner_dict[ner] = total_ner_dict.get(ner, 0) + 1

    for entity in output_entities:
        ner = entity[2]
        total_output_ner_dict[ner] = total_output_ner_dict.get(ner, 0) + 1
 
    for entity in correct_entities:
        ner = entity[2]
        correct_ner_dict[ner] = correct_ner_dict.get(ner, 0) + 1


print(total_ner_dict)
print(total_output_ner_dict)
print(correct_ner_dict)
print("%s|%s|%s|%s" %('Class', "Precision", "Recall", "F1-Score"))
print("--|--|--|--")
for ner in total_ner_dict.keys():
    precision = correct_ner_dict[ner] / total_output_ner_dict[ner]
    recall = correct_ner_dict[ner] / total_ner_dict[ner]
    f1_score = 2 * precision * recall / (precision + recall)
    print("%s|%f|%f|%f" % (ner, precision, recall, f1_score))

overall_precision = sum(correct_ner_dict.values()) / sum(total_output_ner_dict.values())
overall_recall =sum(correct_ner_dict.values()) / sum(total_ner_dict.values()) 
overall_f_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
print("Overall F-score is %f"%overall_f_score)
