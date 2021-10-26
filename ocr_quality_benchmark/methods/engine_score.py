import json
from statistics import mean


def score_file_engine_scores(name: str, path_to_json: str) -> float:
    with open(path_to_json) as file:
        data_json = json.load(file)

    return float(mean(data_json['scores']))
