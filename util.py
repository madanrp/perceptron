import re

def get_word_shape(word):
    word = re.sub("[A-Z]+", "A", re.sub("[a-z]+", "a", word))        
    word = re.sub("[0-9]+", "1", word)
    word = re.sub("[^A-Za-z0-9]+", "_", word)
    return word

def suffix(word):
    if len(word) > 3:
        return word[-3:]
    else:
        return "*N*"

if __name__ == "__main__":
    assert get_word_shape("MadaN") == "AaA"
    assert get_word_shape("MaDaN") == "AaAaA"
    assert get_word_shape("MadNN") == "AaA"
    assert get_word_shape("MADAN") == "A"
    assert get_word_shape("madan") == "a"
    assert get_word_shape("madan123ASD@@@") == "a1A_"
