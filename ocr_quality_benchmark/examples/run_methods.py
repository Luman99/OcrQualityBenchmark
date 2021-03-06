import logging

from ocr_quality_benchmark.Benchmark import Benchmark
from ocr_quality_benchmark.methods.DecisionTreeMethod import DecisionTreeMethod
from ocr_quality_benchmark.methods.DictionaryMethod import DictionaryMethod
from ocr_quality_benchmark.methods.EngineScoreMethod import score_file_engine_scores

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

languages = ['pol']
ocr_engines = ['tesseract']
train_test = ['train', 'test']
train = ['train']
test = ['test']
data_source = ['fiszki_ocr']

benchmark_train = Benchmark(languages=languages, ocr_engines=ocr_engines,
                            train_test=train, data_source=data_source)

benchmark_test = Benchmark(languages=languages, ocr_engines=ocr_engines,
                           train_test=test, data_source=data_source)

benchmark_all = Benchmark(languages=languages, ocr_engines=ocr_engines,
                          train_test=train_test, data_source=data_source)

logging.warning('engine score')
logging.warning(benchmark_test.rate_method(score_file_engine_scores))
benchmark_test.analysis_result(score_file_engine_scores, True)

decision_method = DecisionTreeMethod('../resources/decision_tree_tesseract.joblib')
logging.warning('decision_tree_tesseract')
logging.warning(benchmark_test.rate_method(decision_method.score_file))
benchmark_test.analysis_result(decision_method.score_file, True)

dictionary_method = DictionaryMethod()
logging.warning('dictionary score')
logging.warning(benchmark_test.rate_method(dictionary_method.score_file))
benchmark_test.analysis_result(dictionary_method.score_file, True)
