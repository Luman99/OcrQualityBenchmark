from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from ocr_quality_benchmark.Benchmark import Benchmark
from joblib import dump
from ocr_quality_benchmark.methods.EngineScoreMethod import score_file_engine_scores


def make_model_decision_tree(benchmark: Benchmark) -> None:
    data_train = benchmark.data
    benchmark.rate_method(score_file_engine_scores)
    data_train = data_train[['ocr_quality_wer', 'percent_of_white_spaces', 'number_of_tokens', 'method_ocr_quality']]

    decision_tree_gx = DecisionTreeRegressor(max_depth=4, random_state=1)
    X = data_train[['method_ocr_quality', 'percent_of_white_spaces', 'number_of_tokens']]
    y = data_train['ocr_quality_wer'] - data_train['method_ocr_quality']
    decision_tree_gx = decision_tree_gx.fit(X, y)
    dump(decision_tree_gx, '../resources/decision_tree_tesseract.joblib')


if __name__ == '__main__':
    benchmark_train = Benchmark(languages=['pol'], ocr_engines=['tesseract'],
                                train_test=['train'], data_source=['fiszki_ocr'])
    make_model_decision_tree(benchmark_train)
