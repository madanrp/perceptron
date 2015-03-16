#!/usr/bin/python3
import argparse

def get_train_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", action="store", dest="DEVFILE", default=None)
    parser.add_argument('TRAININGFILE', action="store")
    parser.add_argument('MODELFILE', action="store")
    args = parser.parse_args()
    return args

def get_test_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('MODELFILE', action="store")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
