import json
from statistics import mean


def dictionary_method(name: str, path_to_json: str, alfabet: [str]) -> float:
    # alfabet = []
    # with open("D:\Inżynierka\Projekty\data\\alfabet.txt") as alf:
    #     for line in alf:
    #         alfabet.append(line[:2])

    # name = "81f53b2c4e50191fba69f7c381db5079760f9d62.png"
    text2 = ''
    with open("D:\Inżynierka\Projekty\data\dev-0\in.tsv", encoding='latin1') as file:
        for line in file:
            if name in line:
                text2 = (line[53:])

    good = 0
    all = 0
    for word in text2.split(' '):
        all += 1
        if word in alfabet:
            good += 1

    return float(good/all)
