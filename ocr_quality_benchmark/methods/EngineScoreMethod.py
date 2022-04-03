from pandas import DataFrame


def score_file_engine_scores(name: str, data: DataFrame) -> float:
    return data[data['name'] == name]['tesseract_engine_score'].values[0]
