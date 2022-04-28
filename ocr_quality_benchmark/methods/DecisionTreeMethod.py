from joblib import load
from pandas import DataFrame


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
