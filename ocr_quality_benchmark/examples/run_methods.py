import logging

from ocr_quality_benchmark.Benchmark import Benchmark
from ocr_quality_benchmark.methods.DecisionTreeMethod import DecisionTreeMethod
from ocr_quality_benchmark.methods.DictionaryMethod import dictionary_method
from ocr_quality_benchmark.methods.engine_score import score_file_engine_scores

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

languages = ['eng', 'pol']
ocr_engines = ['tesseract', 'textract_ocr_res', 'ms_read_res']
train_test = ['train', 'test']
data_source = ['ocr-test-challenge']

benchmark = Benchmark(languages=['eng'], ocr_engines=['tesseract', 'textract_ocr_res', 'ms_read_res'],
                      train_test=['train', 'test'], data_source=['ocr-test-challenge'])

benchmark_test = Benchmark(languages=['eng'], ocr_engines=['tesseract', 'textract_ocr_res', 'ms_read_res'],
                           train_test=['test'], data_source=['ocr-test-challenge'])

logging.warning('dictionary score')
logging.warning(benchmark.rate_method(dictionary_method))

# logging.warning('dictionary score')
# logging.warning(benchmark.rate_method(score_file_engine_scores))
# benchmark.analysis_result(score_file_engine_scores, True)
#
# logging.warning('engine score')
# logging.warning(benchmark.rate_method(score_file_engine_scores))
# benchmark.analysis_result(score_file_engine_scores, True)
#
# decision_method = DecisionTreeMethod('../resources/decision_tree_tesseract.joblib')
# logging.warning('decision_tree_tesseract')
# logging.warning(benchmark_test.rate_method(decision_method.score_file))
