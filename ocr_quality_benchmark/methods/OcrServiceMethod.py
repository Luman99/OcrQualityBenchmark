import json
from typing import Optional

from ocr_quality_service.data.ocr_scorer_document import OCRScorerDocument
from ocr_quality_service.scorer.dictionary_based import DictionaryQualityScorer


class OcrServiceMethod:

    def __init__(self) -> None:
        self._dictionary_quality_scorer: DictionaryQualityScorer = DictionaryQualityScorer()

    def score_file(self, name: str, path_to_json: str) -> float:
        with open(path_to_json) as file:
            data_json = json.load(file)

        token_span = []
        position_of_token = 0

        for span in data_json['tokens']:
            token_span.append([position_of_token, position_of_token + len(span)])
            position_of_token = position_of_token + len(span) + 1

        ocr_scorer_document = OCRScorerDocument(
            document_id=name, token_bounding_box=data_json['positions'],
            token_text=data_json['tokens'], token_span=token_span,
            page_span=[], page_bounding_box=[],
            token_score=data_json['scores'])

        return float(self._dictionary_quality_scorer.score_document(ocr_scorer_document))

