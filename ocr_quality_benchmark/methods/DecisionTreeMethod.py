import json
from statistics import mean
import pandas as pd

from joblib import load
from pandas import DataFrame

from ocr_quality_benchmark.methods.Constants import dataset
from ocr_quality_benchmark.methods.EngineScoreMethod import score_file_engine_scores


class DecisionTreeMethod:
    def __init__(self, model: str) -> None:
        self._model = load(model)

    def score_file(self, name: str, data: DataFrame) -> float:
        result = DataFrame()

        result['method_ocr_quality'] = data[data['name'] == name]['tesseract_engine_score']
        result['percent_of_white_spaces'] = data[data['name'] == name]['percent_of_white_spaces']
        result['number_of_tokens'] = data[data['name'] == name]['number_of_tokens']

        score = round(float(self._model.predict(result)[0] +
                           data[data['name'] == name]['tesseract_engine_score'].values[0]), 2)
        if score > 1:
            score = 1
        if score < 0:
            score = 0

        return score

# class DecisionTreeMethod:
#     def __init__(self, model: str) -> None:
#         self._model = load(model)
#
#     def score_file(self, name: str, path_to_json: str) -> float:
#         results = score_file_engine_scores_for_decision_tree(name=name, path_to_json=path_to_json)
#         data = DataFrame()
#
#         data['method_ocr_quality'] = [results[0]]
#         data['percent_white_spaces'] = [results[1]]
#         data['number_of_tokens'] = [results[2]]
#         return float(self._model.predict(data)[0] + results[0])
