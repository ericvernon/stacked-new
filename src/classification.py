import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from lib import DIFFICULTY_EASY, DIFFICULTY_HARD, DIFFICULTY_VERY_HARD, SENTINEL_REJECT

logger = logging.getLogger(__name__)


@dataclass
class Results:
    # Overall accuracy, considering all patterns
    white_box_accuracy: float
    black_box_accuracy: float
    combined_accuracy: float

    # Accuracy only on patterns graded as "easy"
    white_box_accuracy_easy: float
    black_box_accuracy_easy: float
    combined_accuracy_easy: float

    # Accuracy only on patterns graded as "hard"
    white_box_accuracy_hard: float
    black_box_accuracy_hard: float
    combined_accuracy_hard: float

    # Percentage of patterns graded as "hard"
    deferral_rate: float


@dataclass
class ResultsTernary(Results):
    # Accuracy only on patterns graded as "very hard"
    white_box_accuracy_very_hard: float
    black_box_accuracy_very_hard: float
    combined_accuracy_very_hard: float

    # Percentage of patterns graded as "very hard"
    reject_rate: float




def calculate_results_binary_grader(X, y_truth, white_box_model, black_box_model, grader_model):
    grades = grader_model.predict(X)
    easy_idx = grades == DIFFICULTY_EASY
    hard_idx = grades == DIFFICULTY_HARD

    white_box_predictions = white_box_model.predict(X)
    black_box_predictions = black_box_model.predict(X)

    final_predictions = white_box_predictions.copy()
    final_predictions[hard_idx] = black_box_predictions[hard_idx]

    n_patterns = X.shape[0]
    n_easy = np.count_nonzero(easy_idx)
    n_hard = np.count_nonzero(hard_idx)

    white_box_accuracy = np.count_nonzero(white_box_predictions == y_truth) / n_patterns
    white_box_accuracy_easy = (np.count_nonzero(white_box_predictions[easy_idx] == y_truth[easy_idx]) / n_easy) if n_easy > 0 else 0
    white_box_accuracy_hard = (np.count_nonzero(white_box_predictions[hard_idx] == y_truth[hard_idx]) / n_hard) if n_hard > 0 else 0

    black_box_accuracy = np.count_nonzero(black_box_predictions == y_truth) / n_patterns
    black_box_accuracy_easy = (np.count_nonzero(black_box_predictions[easy_idx] == y_truth[easy_idx]) / n_easy) if n_easy > 0 else 0
    black_box_accuracy_hard = (np.count_nonzero(black_box_predictions[hard_idx] == y_truth[hard_idx]) / n_hard) if n_hard > 0 else 0

    final_accuracy = np.count_nonzero(final_predictions == y_truth) / n_patterns
    final_accuracy_easy = (np.count_nonzero(final_predictions[easy_idx] == y_truth[easy_idx]) / n_easy) if n_easy > 0 else 0
    final_accuracy_hard = (np.count_nonzero(final_predictions[hard_idx] == y_truth[hard_idx]) / n_hard) if n_hard > 0 else 0

    deferral_rate = n_hard / n_patterns

    results = Results(
        white_box_accuracy=white_box_accuracy,
        white_box_accuracy_easy=white_box_accuracy_easy,
        white_box_accuracy_hard=white_box_accuracy_hard,
        black_box_accuracy=black_box_accuracy,
        black_box_accuracy_easy=black_box_accuracy_easy,
        black_box_accuracy_hard=black_box_accuracy_hard,
        combined_accuracy=final_accuracy,
        combined_accuracy_easy=final_accuracy_easy,
        combined_accuracy_hard=final_accuracy_hard,
        deferral_rate=deferral_rate,
    )

    return results

def calculate_results_ternary_grader(X, y_truth, white_box_model, black_box_model, grader_model):
    grades = grader_model.predict(X)
    easy_idx = grades == DIFFICULTY_EASY
    hard_idx = grades == DIFFICULTY_HARD
    very_hard_idx = grades == DIFFICULTY_VERY_HARD

    white_box_predictions = white_box_model.predict(X)
    black_box_predictions = black_box_model.predict(X)

    final_predictions = white_box_predictions.copy()
    final_predictions[hard_idx] = black_box_predictions[hard_idx]
    final_predictions[very_hard_idx] = SENTINEL_REJECT

    n_patterns = X.shape[0]
    n_easy = np.count_nonzero(easy_idx)
    n_hard = np.count_nonzero(hard_idx)
    n_very_hard = np.count_nonzero(very_hard_idx)

    white_box_accuracy = np.count_nonzero(white_box_predictions == y_truth) / n_patterns
    white_box_accuracy_easy = (np.count_nonzero(white_box_predictions[easy_idx] == y_truth[easy_idx]) / n_easy) if n_easy > 0 else 0
    white_box_accuracy_hard = (np.count_nonzero(white_box_predictions[hard_idx] == y_truth[hard_idx]) / n_hard) if n_hard > 0 else 0
    white_box_accuracy_very_hard = (np.count_nonzero(white_box_predictions[very_hard_idx] == y_truth[very_hard_idx]) / n_very_hard) if n_very_hard > 0 else 0

    black_box_accuracy = np.count_nonzero(black_box_predictions == y_truth) / n_patterns
    black_box_accuracy_easy = (np.count_nonzero(black_box_predictions[easy_idx] == y_truth[easy_idx]) / n_easy) if n_easy > 0 else 0
    black_box_accuracy_hard = (np.count_nonzero(black_box_predictions[hard_idx] == y_truth[hard_idx]) / n_hard) if n_hard > 0 else 0
    black_box_accuracy_very_hard = (np.count_nonzero(black_box_predictions[very_hard_idx] == y_truth[very_hard_idx]) / n_very_hard) if n_very_hard > 0 else 0

    #
    final_accuracy = np.count_nonzero(final_predictions == y_truth) / (n_patterns - n_very_hard)
    final_accuracy_easy = (np.count_nonzero(final_predictions[easy_idx] == y_truth[easy_idx]) / n_easy) if n_easy > 0 else 0
    final_accuracy_hard = (np.count_nonzero(final_predictions[hard_idx] == y_truth[hard_idx]) / n_hard) if n_hard > 0 else 0
    final_accuracy_very_hard = (np.count_nonzero(final_predictions[very_hard_idx] == y_truth[very_hard_idx]) / n_very_hard) if n_very_hard > 0 else 0

    deferral_rate = n_hard / n_patterns
    reject_rate = n_very_hard / n_patterns

    results = ResultsTernary(
        white_box_accuracy=white_box_accuracy,
        white_box_accuracy_easy=white_box_accuracy_easy,
        white_box_accuracy_hard=white_box_accuracy_hard,
        white_box_accuracy_very_hard=white_box_accuracy_very_hard,
        black_box_accuracy=black_box_accuracy,
        black_box_accuracy_easy=black_box_accuracy_easy,
        black_box_accuracy_hard=black_box_accuracy_hard,
        black_box_accuracy_very_hard=black_box_accuracy_very_hard,
        combined_accuracy=final_accuracy,
        combined_accuracy_easy=final_accuracy_easy,
        combined_accuracy_hard=final_accuracy_hard,
        combined_accuracy_very_hard=final_accuracy_very_hard,
        deferral_rate=deferral_rate,
        reject_rate=reject_rate,
    )

    return results

def write_header_binary_grader(f):
    f.write('split,')
    f.write('white_box_name,black_box_name,grader_name,')
    f.write('white_box_accuracy_total,white_box_accuracy_easy,white_box_accuracy_hard,')
    f.write('black_box_accuracy_total,black_box_accuracy_easy,black_box_accuracy_hard,')
    f.write('final_accuracy_total,final_accuracy_easy,final_accuracy_hard,')
    f.write('deferral_rate')
    f.write('\n')


def save_results_binary_grader(fh, split_idx, white_box_name, black_box_name, grader_name, results: Results):
    fh.write(f'{split_idx},')
    fh.write(f'{white_box_name},{black_box_name},{grader_name},')
    fh.write(f'{results.white_box_accuracy},{results.white_box_accuracy_easy},{results.white_box_accuracy_hard},')
    fh.write(f'{results.black_box_accuracy},{results.black_box_accuracy_easy},{results.black_box_accuracy_hard},')
    fh.write(f'{results.combined_accuracy},{results.combined_accuracy_easy},{results.combined_accuracy_hard},')
    fh.write(f'{results.deferral_rate}')
    fh.write('\n')


def write_header_ternary_grader(f):
    f.write('split,')
    f.write('white_box_name,black_box_name,grader_name,')
    f.write('white_box_accuracy_total,white_box_accuracy_easy,white_box_accuracy_hard,white_box_accuracy_very_hard,')
    f.write('black_box_accuracy_total,black_box_accuracy_easy,black_box_accuracy_hard,black_box_accuracy_very_hard,')
    f.write('final_accuracy_total,final_accuracy_easy,final_accuracy_hard,final_accuracy_very_hard,')
    f.write('deferral_rate,reject_rate')
    f.write('\n')


def save_results_ternary_grader(fh, split_idx, white_box_name, black_box_name, grader_name, results: ResultsTernary):
    fh.write(f'{split_idx},')
    fh.write(f'{white_box_name},{black_box_name},{grader_name},')
    fh.write(f'{results.white_box_accuracy},{results.white_box_accuracy_easy},{results.white_box_accuracy_hard},{results.white_box_accuracy_very_hard},')
    fh.write(f'{results.black_box_accuracy},{results.black_box_accuracy_easy},{results.black_box_accuracy_hard},{results.black_box_accuracy_very_hard},')
    fh.write(f'{results.combined_accuracy},{results.combined_accuracy_easy},{results.combined_accuracy_hard},{results.combined_accuracy_very_hard},')
    fh.write(f'{results.deferral_rate},{results.reject_rate}')
    fh.write('\n')


def save_results(path, X, y_truth, glass_box_model, black_box_model, grader_model, save_X=False):
    n_features = X.shape[1]
    if save_X:
        df = pd.DataFrame(X, index=None, columns=[f'X_{i}' for i in range(n_features)])
    else:
        df = pd.DataFrame()
    df['y_truth'] = y_truth
    df['y_glass'] = glass_box_model.predict(X)
    df['y_black'] = black_box_model.predict(X)
    df['y_grader'] = grader_model.predict(X)
    df.to_csv(path, index=False)


RESULTS_SUMMARY_HEADER = ("data_split,white_box_model,black_box_model,grader_model,grader_type,"
                          "white_box_accuracy_train,white_box_accuracy_test,"
                          "black_box_accuracy_train,black_box_accuracy_test,"
                          "combined_accuracy_train,combined_accuracy_test,"
                          "deferral_rate_train,deferral_rate_test,"
                          "reject_rate_train,reject_rate_test,"
                          "white_box_accuracy_train_easy", "white_box_accuracy_test"
                                                           "")
