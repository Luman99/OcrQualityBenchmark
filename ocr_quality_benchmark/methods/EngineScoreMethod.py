import json
from statistics import mean
import pandas as pd

from ocr_quality_benchmark.methods.Constants import dataset


def score_file_engine_scores(name: str) -> float:
    #print(name)
    score = 0

    data = pd.read_csv(dataset)
    #print(data)
    for row in data.itertuples():
        if row[1] == name:
            score = row[3]
    # with open(dataset) as file:
    #     print(file[0])
    #     if file(name) == name:
    #             score = file[name]
    return float(score)
