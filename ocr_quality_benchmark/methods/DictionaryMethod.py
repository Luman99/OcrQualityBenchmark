from pandas import DataFrame

from ocr_quality_benchmark.methods.Constants import alphabet


def dictionary_method(name: str, data: DataFrame) -> float:
    text2 = ''
    with open("D:\Inzynierka\Projekty\data\dev-0\in.tsv", encoding='latin1') as file:
        for line in file:
            if name in line:
                text2 = (line[53:])
                # print(text2)

    correct_words = 0
    all_words = 0
    with open(alphabet) as alp:
        alp = alp.readlines()
        for word in text2.split(' '):
            all_words += 1
            # print(word)
            # print(alp)
            if word in alp:
                correct_words += 1
    #print(correct_words)

    return float(correct_words/all_words)
