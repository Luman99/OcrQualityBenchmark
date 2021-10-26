import json
from statistics import mean
from typing import List
from sklearn.tree import DecisionTreeRegressor

from ocr_quality_benchmark.Benchmark import Benchmark
from joblib import dump


def score_file_engine_scores_for_decision_tree(name: str, path_to_json: str) -> List[int]:
    with open(path_to_json) as file:
        data_json = json.load(file)

    return [mean(data_json['scores']), len([word for word in data_json['tokens'] if word.isspace()]),
            len(data_json['tokens'])]


def make_model_decision_tree(benchmark: Benchmark) -> None:
    data_train = benchmark.get_data(score_file_engine_scores_for_decision_tree)
    data_train = data_train[['gold_ocr_quality_wer', 'percent_white_spaces', 'number_of_tokens', 'method_ocr_quality']]

    decision_tree_gx = DecisionTreeRegressor(max_depth=5)
    X = data_train[['method_ocr_quality', 'percent_white_spaces', 'number_of_tokens']]
    y = data_train['gold_ocr_quality_wer'] - data_train['method_ocr_quality']
    decision_tree_gx = decision_tree_gx.fit(X, y)
    dump(decision_tree_gx, '../resources/decision_tree_tesseract.joblib')


if __name__ == '__main__':
    benchmark_train = Benchmark(languages=['eng'], ocr_engines=['tesseract'],
                                train_test=['train'], data_source=['ocr-test-challenge'])
    make_model_decision_tree(benchmark_train)
