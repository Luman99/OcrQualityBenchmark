import json
from statistics import mean

from joblib import load
from pandas import DataFrame

from ocr_quality_benchmark.methods.EngineScoreMethod import score_file_engine_scores
from ocr_quality_benchmark.utils.make_models import score_file_engine_scores_for_decision_tree


class DecisionTreeMethod:
    def __init__(self, model: str) -> None:
        self._model = load(model)

    def score_file(self, name: str, path_to_json: str) -> float:
        results = score_file_engine_scores_for_decision_tree(name=name, path_to_json=path_to_json)
        data = DataFrame()

        data['method_ocr_quality'] = [results[0]]
        data['percent_white_spaces'] = [results[1]]
        data['number_of_tokens'] = [results[2]]
        return float(self._model.predict(data)[0] + results[0])
