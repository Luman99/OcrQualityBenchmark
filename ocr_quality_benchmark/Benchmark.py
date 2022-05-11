import logging
import time
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
        self.data = self._prepare_gold_data()

    def _calculate_scores_from_method(self, method) -> None:
        ocr_qualities = []
        for row in self.data.itertuples():
            value = method(name=row[1], data=self.data)
            ocr_qualities.append(value)

        if 'method_ocr_quality' in self.data.columns:
            self.data = self.data.drop(columns=['method_ocr_quality'])
        self.data['method_ocr_quality'] = ocr_qualities

    def _prepare_gold_data(self) -> pd.DataFrame:
        if self._path_to_csv is None:
            data = pd.read_csv(dataset)
        else:
            data = pd.read_csv(self._path_to_csv)
        data = data.loc[(data['language'].isin(self._languages)) &
                        (data['data_source'].isin(self._data_source)) &
                        (data['ocr_engine'].isin(self._ocr_engines)) &
                        (data['train_test'].isin(self._train_test))]

        return data

    def analysis_result(self, method, show_plots: bool = False):
        if 'method_ocr_quality' not in self.data.columns:
            self._calculate_scores_from_method(method)

        logging.warning(f"mean for gold - {self.data['ocr_quality_wer'].mean()}")
        logging.warning(f"mean for method - {self.data['method_ocr_quality'].mean()}")

        if show_plots:
            self.make_plots()

    def make_plots(self):
        abs_data = (self.data['ocr_quality_wer'] - self.data['method_ocr_quality']).abs()

        bins = np.histogram(np.hstack((self.data['ocr_quality_wer'])),
                            bins=50)[1]

        fig, axs = plt.subplots(2, 2)

        axs[0, 0].set_title('Histograms')
        axs[0, 0].hist(self.data['ocr_quality_wer'], alpha=0.5, color='red', bins=bins, label='gold')
        axs[0, 0].hist(self.data['method_ocr_quality'], alpha=0.5, color='blue', bins=bins, label='method')
        axs[0, 0].legend()
        axs[0, 0].set_ylim(0, 30)

        axs[0, 1].set_title('Histogram abs')
        abs_data.hist(alpha=0.5, color='green', bins=bins, label='abs_values', ax=axs[0, 1])
        axs[0, 1].set_ylim(0, 30)

        axs[1, 0].plot(self.data['ocr_quality_wer'], self.data['method_ocr_quality'], 'ro')
        axs[1, 0].set_xlabel('Wer')
        axs[1, 0].set_ylabel('Method')
        axs[1, 0].set_xlim(0, 1)
        axs[1, 0].set_ylim(0, 1)
        axs[1, 0].set_aspect('equal')
        plt.show()

        # f, a = plt.subplots(2, 5)
        # a = a.ravel()
        # f.suptitle("WER where score_document > k")
        # for x, ax in enumerate(a):
        #     x /= 10
        #     ax.hist(self.data.loc[self.data['method_ocr_quality'] > x, 'ocr_quality_wer'], range=(0, 1))
        #     ax.set_title(f'k = {x}')
        #     ax.set_xlabel('WER')
        #     ax.set_ylabel('number of files')
        #     ax.set_ylim(0, 120)
        # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        #
        # f, a = plt.subplots(2, 5)
        # a = a.ravel()
        # f.suptitle("WER where score_document <= k")
        # for x, ax in enumerate(a):
        #     x /= 10
        #     ax.hist(self.data.loc[self.data['method_ocr_quality'] <= x, 'ocr_quality_wer'], range=(0, 1))
        #     ax.set_title(f'k = {x}')
        #     ax.set_xlabel('WER')
        #     ax.set_ylabel('number of files')
        #     ax.set_ylim(0, 120)
        # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        # plt.show()

    def rate_method(self, method) -> Dict[str, float]:
        time_start = time.time()
        self._calculate_scores_from_method(method)
        time_end = time.time()
        logging.warning(f'Evaluate this method took {time_end-time_start} seconds')
        return {'mean_squared_error': mean_squared_error(y_true=self.data['ocr_quality_wer'],
                                                         y_pred=self.data['method_ocr_quality']),
                'mean_absolute_error': mean_absolute_error(y_true=self.data['ocr_quality_wer'],
                                                           y_pred=self.data['method_ocr_quality'])}
