import re

from pandas import DataFrame
from string import punctuation
from ocr_quality_benchmark.methods.Constants import polish_dictionary

regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')


def load_dictionary():
    all_words = {'a'}
    with open(polish_dictionary, encoding='utf8') as alp:
        alp = alp.readlines()
        for w in alp:
            w = w.replace(' ', '')
            w = w.replace('\n', '')
            for a in w.split(','):
                all_words.add(a)
    return all_words


class DictionaryMethod:
    def __init__(self):
        self.polish_dictionary = load_dictionary()

    def score_file(self, name: str, data: DataFrame) -> float:
        text = ''
        with open("..\..\..\data\dev-0\in.tsv", encoding='utf-8') as file:
            for line in file:
                if name in line:
                    text = (line[53:])

        correct_words = 0
        all_words = 0
        texts = text.split(' ')
        for temp_word in texts:
            words = temp_word.split('\\n')
            for word in words:
                word = word.lower()
                word = word.lstrip(punctuation)
                word = word.rstrip(punctuation)

                if not word.isnumeric() and not re.fullmatch(regex, word) and word != '\n' and word not in punctuation \
                        and word != '':
                    all_words += 1
                    if word in self.polish_dictionary:
                        if len(word) != 1 or word in ['a', 'i', 'o', 'u', 'w', 'z']:
                            correct_words += 1
        if all_words == 0:
            result = 0
        else:
            result = float(correct_words/all_words)
        return result
