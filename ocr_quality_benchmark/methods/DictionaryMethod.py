import re

from pandas import DataFrame
from string import punctuation
from ocr_quality_benchmark.methods.Constants import alphabet

regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')


class DictionaryMethod:
    def __init__(self):
        self.alphabet = self._load_alphabet()

    def _load_alphabet(self):
        all_words = {'a'}
        with open(alphabet, encoding='utf8') as alp:
            alp = alp.readlines()
            for w in alp:
                w = w.replace(' ', '')
                w = w.replace('\n', '')
                for a in w.split(','):
                    all_words.add(a)
        return all_words

    def score_file(self, name: str, data: DataFrame) -> float:
        text2 = ''
        with open("D:\Inzynierka\Projekty\data\dev-0\in.tsv", encoding='utf-8') as file:
            for line in file:
                if name in line:
                    text2 = (line[53:])

        correct_words = 0
        all_words = 0
        texts = text2.split(' ')
        for temp_word in texts:
            words = temp_word.split('\\n')
            for word in words:
                word = word.lower()

                word = word.lstrip(punctuation)
                word = word.rstrip(punctuation)

                word = word.lstrip('\\n')
                word = word.rstrip('\\n')

                #print(f'\n{word}')
                if not word.isnumeric() and not re.fullmatch(regex, word) and word != '\n' and word not in punctuation:
                    all_words += 1
                    #print('word')
                    if word in self.alphabet:
                        correct_words += 1
                        #print('correct')
        if all_words == 0:
            result = 0
        else:
            result = float(correct_words/all_words)
        return result
