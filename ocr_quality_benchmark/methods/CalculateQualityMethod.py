import pandas as pd

from ocr_quality_benchmark.methods.Constants import PATH_DATA


class CalculateQualityMethod:

    def __init__(self) -> None:
        self._data = self._prepare_data_to_test()

    @staticmethod
    def _prepare_data_to_test():
        data_to_test = pd.read_csv(f'{PATH_DATA}/ocr_quality_for_calculate_quality_method.txt',
                                   sep=' ', names=['name', 'ocr_quality'])
        data_to_test.insert(1, 'ocr_engine', 'tesseract')
        return data_to_test

    def score_file(self, name: str, path_to_json: str) -> float:
        value = self._data.loc[(self._data['name'] == name) & (self._data['ocr_engine'] == 'tesseract')]['ocr_quality']
        if len(value) != 0:
            return float(value.values[0])
        return 0.0
