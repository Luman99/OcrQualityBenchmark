import logging
from datetime import datetime
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ocr_quality_benchmark.methods.Constants import dataset
import numpy as np
import seaborn as sns


class Benchmark:
    LOG = logging.getLogger()

    def __init__(self, languages: List[str], ocr_engines: List[str],
                 train_test: List[str], data_source: List[str], path_to_csv: Optional[str] = None) -> None:
        self._data_source = data_source
        self._train_test = train_test
        self._ocr_engines = ocr_engines
        self._languages = languages
        self._path_to_csv = path_to_csv
        self._start_time: datetime = None  # type: ignore
        self._data = self._prepare_gold_data()
        # self.alphabet = self._prepare_alphabet()

    # def _prepare_alphabet(self):
    #     alfabet = []
    #     with open("D:\Inżynierka\Projekty\data\\alfabet.txt") as alf:
    #         for line in alf:
    #             alfabet.append(line[:2])
    #     return alfabet

    def _calculate_scores_from_method(self, method, *parameters) -> None:
        ocr_qualities = []
        for row in self._data.itertuples():
            #print(row)
            #print('xddd')
            if len(parameters) == 0:
                value = method(name=row[1])
            else:
                value = method(name=row[1], parameters=parameters)
            ocr_qualities.append(value)

        if 'method_ocr_quality' in self._data.columns:
            self._data = self._data.drop(columns=['method_ocr_quality'])
        self._data['method_ocr_quality'] = ocr_qualities

    def _prepare_test_data_for_decision_tree(self, method) -> None:
        ocr_qualities = []
        white_spaces = []
        number_of_tokens = []
        for file_dictionary in self._data.itertuples():
            ocr_qualities.append(method(name=file_dictionary[1], path_to_json=file_dictionary[5])[0])
            white_spaces.append(method(name=file_dictionary[1], path_to_json=file_dictionary[5])[1])
            number_of_tokens.append(method(name=file_dictionary[1], path_to_json=file_dictionary[5])[2])

        if 'method_ocr_quality' in self._data.columns:
            self._data = self._data.drop(columns=['method_ocr_quality'])

        self._data['method_ocr_quality'] = ocr_qualities
        self._data['percent_white_spaces'] = white_spaces
        self._data['number_of_tokens'] = number_of_tokens

    def _prepare_gold_data(self) -> pd.DataFrame:
        if self._path_to_csv is None:
            data = pd.read_csv(dataset)
        else:
            data = pd.read_csv(self._path_to_csv)
        data = data.loc[(data['language'].isin(self._languages)) &
                        (data['data_source'].isin(self._data_source)) &
                        (data['ocr_engine'].isin(self._ocr_engines)) &
                        (data['train_test'].isin(self._train_test))]

        data = data.rename(columns={'ocr_quality_wer': 'gold_ocr_quality_wer'})
        return data

    def analysis_result(self, method, show_plots: bool = False):
        if 'method_ocr_quality' not in self._data.columns:
            self._calculate_scores_from_method(method)

        logging.warning(f"mean for gold - {self._data['gold_ocr_quality_wer'].mean()}")
        logging.warning(f"mean for method - {self._data['method_ocr_quality'].mean()}")

        if show_plots:
            self.make_plots()

    def make_plots(self):
        correlations = self._data.corr().copy()
        correlations = correlations.rename(columns={'gold_ocr_quality_wer': 'Wer', 'gold_ocr_quality_cer': 'Cer',
                                                    'gold_ocr_quality_iou': 'Iou', 'method_ocr_quality': 'Method'})

        abs_data = (self._data['gold_ocr_quality_wer'] - self._data['method_ocr_quality']).abs()

        bins = np.histogram(np.hstack((self._data['gold_ocr_quality_wer'])),
                            bins=50)[1]

        fig, axs = plt.subplots(2, 2)

        axs[0, 0].set_title('Histograms')
        axs[0, 0].hist(self._data['gold_ocr_quality_wer'], alpha=0.5, color='red', bins=bins, label='gold')
        axs[0, 0].hist(self._data['method_ocr_quality'], alpha=0.5, color='blue', bins=bins, label='method')
        axs[0, 0].legend()
        axs[0, 0].set_ylim(0, 30)

        axs[0, 1].set_title('Histogram abs')
        abs_data.hist(alpha=0.5, color='green', bins=bins, label='abs_values', ax=axs[0, 1])
        axs[0, 1].set_ylim(0, 30)

        axs[1, 0].set_title('Correlations')
        sns.heatmap(correlations, cmap='coolwarm', vmin=0.4, vmax=1, center=0, annot=True,
                    square=True, linewidths=.5, ax=axs[1, 0])

        axs[1, 1].plot(self._data['gold_ocr_quality_wer'], self._data['method_ocr_quality'], 'ro')
        axs[1, 1].set_xlabel('Wer')
        axs[1, 1].set_ylabel('Method')
        axs[1, 1].set_xlim(0, 1)
        axs[1, 1].set_ylim(0, 1)
        axs[1, 1].set_aspect('equal')
        plt.show()

        f, a = plt.subplots(2, 5)
        a = a.ravel()
        f.suptitle("WER where score_document > k")
        for x, ax in enumerate(a):
            x /= 10
            ax.hist(self._data.loc[self._data['method_ocr_quality'] > x, 'gold_ocr_quality_wer'], range=(0, 1))
            ax.set_title(f'k = {x}')
            ax.set_xlabel('WER')
            ax.set_ylabel('number of files')
            ax.set_ylim(0, 120)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        f, a = plt.subplots(2, 5)
        a = a.ravel()
        f.suptitle("WER where score_document <= k")
        for x, ax in enumerate(a):
            x /= 10
            ax.hist(self._data.loc[self._data['method_ocr_quality'] <= x, 'gold_ocr_quality_wer'], range=(0, 1))
            ax.set_title(f'k = {x}')
            ax.set_xlabel('WER')
            ax.set_ylabel('number of files')
            ax.set_ylim(0, 120)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()

    def rate_method(self, method) -> Dict[str, float]:
        self._calculate_scores_from_method(method)
        return {'mean_squared_error': mean_squared_error(y_true=self._data['gold_ocr_quality_wer'],
                                                         y_pred=self._data['method_ocr_quality']),
                'mean_absolute_error': mean_absolute_error(y_true=self._data['gold_ocr_quality_wer'],
                                                           y_pred=self._data['method_ocr_quality'])}

    def get_data(self, method):
        self._prepare_test_data_for_decision_tree(method)
        return self._data
