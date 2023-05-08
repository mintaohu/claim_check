import string


def filter_english_character(sentence):
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, sentence))