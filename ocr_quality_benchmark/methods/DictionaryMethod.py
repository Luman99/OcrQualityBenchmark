import re

from pandas import DataFrame

from ocr_quality_benchmark.methods.Constants import alphabet

regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')

def dictionary_method(name: str, data: DataFrame) -> float:
    text2 = ''
    with open("D:\Inzynierka\Projekty\data\dev-0\in.tsv", encoding='utf-8') as file:
        for line in file:
            if name in line:
                text2 = (line[53:])

    correct_words = 0
    all_words = 0

    with open(alphabet) as alp:
        alp = alp.readlines()
        alp = [w[:-2] for w in alp]
        #print(alp)
        for word in text2.split(' '):
            word = word.lower()
            #if not word.isspace():
            all_words += 1
            #print(word)
            if word.isnumeric() or re.fullmatch(regex, word) or word in alp:
                correct_words += 1
                #print('correct')

    return float(correct_words/all_words)
