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
    white_box_mse: float
    black_box_mse: float
    combined_mse: float

    # Accuracy only on patterns graded as "easy"
    white_box_mse_easy: float
    black_box_mse_easy: float
    combined_mse_easy: float

    # Accuracy only on patterns graded as "hard"
    white_box_mse_hard: float
    black_box_mse_hard: float
    combined_mse_hard: float

    # Percentage of patterns graded as "hard"
    deferral_rate: float


@dataclass
class ResultsTernary(Results):
    # Accuracy only on patterns graded as "very hard"
    white_box_mse_very_hard: float
    black_box_mse_very_hard: float
    combined_mse_very_hard: float

    # Percentage of patterns graded as "very hard"
    reject_rate: float


def calculate_results_binary_grader(X, y_truth, white_box_model, black_box_model, grader_model):
    grades = grader_model.predict(X)
    easy_idx = grades == DIFFICULTY_EASY
    hard_idx = grades == DIFFICULTY_HARD

    white_box_predictions = white_box_model.predict(X)
    white_box_sq_error = np.square(white_box_predictions - y_truth)

    black_box_predictions = black_box_model.predict(X)
    black_box_sq_error = np.square(black_box_predictions - y_truth)

    final_predictions = white_box_predictions.copy()
    final_predictions[hard_idx] = black_box_predictions[hard_idx]
    final_sq_error = np.square(final_predictions - y_truth)

    n_patterns = X.shape[0]
    n_easy = np.count_nonzero(easy_idx)
    n_hard = np.count_nonzero(hard_idx)

    white_box_mse = np.mean(white_box_sq_error)
    white_box_mse_easy = np.sum(white_box_sq_error[easy_idx]) / n_easy if n_easy > 0 else 0
    white_box_mse_hard = np.sum(white_box_sq_error[hard_idx]) / n_hard if n_hard > 0 else 0

    black_box_mse = np.mean(black_box_sq_error)
    black_box_mse_easy = np.sum(black_box_sq_error[easy_idx]) / n_easy if n_easy > 0 else 0
    black_box_mse_hard = np.sum(black_box_sq_error[hard_idx]) / n_hard if n_hard > 0 else 0

    final_mse = np.mean(final_sq_error)
    final_mse_easy = np.sum(final_sq_error[easy_idx]) / n_easy if n_easy > 0 else 0
    final_mse_hard = np.sum(final_sq_error[hard_idx]) / n_hard if n_hard > 0 else 0

    deferral_rate = n_hard / n_patterns

    results = Results(
        white_box_mse=white_box_mse,
        white_box_mse_easy=white_box_mse_easy,
        white_box_mse_hard=white_box_mse_hard,
        black_box_mse=black_box_mse,
        black_box_mse_easy=black_box_mse_easy,
        black_box_mse_hard=black_box_mse_hard,
        combined_mse=final_mse,
        combined_mse_easy=final_mse_easy,
        combined_mse_hard=final_mse_hard,
        deferral_rate=deferral_rate,
    )

    return results

def calculate_results_ternary_grader(X, y_truth, white_box_model, black_box_model, grader_model):
    grades = grader_model.predict(X)
    easy_idx = grades == DIFFICULTY_EASY
    hard_idx = grades == DIFFICULTY_HARD
    very_hard_idx = grades == DIFFICULTY_VERY_HARD

    white_box_predictions = white_box_model.predict(X)
    white_box_sq_error = np.square(white_box_predictions - y_truth)

    black_box_predictions = black_box_model.predict(X)
    black_box_sq_error = np.square(black_box_predictions - y_truth)

    final_predictions = white_box_predictions.copy()
    final_predictions[hard_idx] = black_box_predictions[hard_idx]
    final_predictions[very_hard_idx] = SENTINEL_REJECT
    final_sq_error = np.square(final_predictions - y_truth)

    n_patterns = X.shape[0]
    n_easy = np.count_nonzero(easy_idx)
    n_hard = np.count_nonzero(hard_idx)
    n_very_hard = np.count_nonzero(very_hard_idx)

    white_box_mse = np.mean(white_box_sq_error)
    white_box_mse_easy = np.sum(white_box_sq_error[easy_idx]) / n_easy if n_easy > 0 else 0
    white_box_mse_hard = np.sum(white_box_sq_error[hard_idx]) / n_hard if n_hard > 0 else 0
    white_box_mse_very_hard = np.sum(white_box_sq_error[very_hard_idx]) / n_very_hard if n_very_hard > 0 else 0

    black_box_mse = np.mean(black_box_sq_error)
    black_box_mse_easy = np.sum(black_box_sq_error[easy_idx]) / n_easy if n_easy > 0 else 0
    black_box_mse_hard = np.sum(black_box_sq_error[hard_idx]) / n_hard if n_hard > 0 else 0
    black_box_mse_very_hard = np.sum(black_box_sq_error[very_hard_idx]) / n_very_hard if n_very_hard > 0 else 0

    final_mse = np.mean(final_sq_error)
    final_mse_easy = np.sum(final_sq_error[easy_idx]) / n_easy if n_easy > 0 else 0
    final_mse_hard = np.sum(final_sq_error[hard_idx]) / n_hard if n_hard > 0 else 0
    final_mse_very_hard = np.sum(final_sq_error[very_hard_idx]) / n_very_hard if n_very_hard > 0 else 0

    deferral_rate = n_hard / n_patterns
    reject_rate = n_very_hard / n_patterns

    results = ResultsTernary(
        white_box_mse=white_box_mse,
        white_box_mse_easy=white_box_mse_easy,
        white_box_mse_hard=white_box_mse_hard,
        white_box_mse_very_hard=white_box_mse_very_hard,
        black_box_mse=black_box_mse,
        black_box_mse_easy=black_box_mse_easy,
        black_box_mse_hard=black_box_mse_hard,
        black_box_mse_very_hard=black_box_mse_very_hard,
        combined_mse=final_mse,
        combined_mse_easy=final_mse_easy,
        combined_mse_hard=final_mse_hard,
        combined_mse_very_hard=final_mse_very_hard,
        deferral_rate=deferral_rate,
        reject_rate=reject_rate,
    )

    return results

def write_header_binary_grader(f):
    f.write('split,')
    f.write('white_box_name,black_box_name,grader_name,')
    f.write('white_box_mse_total,white_box_mse_easy,white_box_mse_hard,')
    f.write('black_box_mse_total,black_box_mse_easy,black_box_mse_hard,')
    f.write('final_mse_total,final_mse_easy,final_mse_hard,')
    f.write('deferral_rate')
    f.write('\n')


def save_results_binary_grader(fh, split_idx, white_box_name, black_box_name, grader_name, results: Results):
    fh.write(f'{split_idx},')
    fh.write(f'{white_box_name},{black_box_name},{grader_name},')
    fh.write(f'{results.white_box_mse},{results.white_box_mse_easy},{results.white_box_mse_hard},')
    fh.write(f'{results.black_box_mse},{results.black_box_mse_easy},{results.black_box_mse_hard},')
    fh.write(f'{results.combined_mse},{results.combined_mse_easy},{results.combined_mse_hard},')
    fh.write(f'{results.deferral_rate}')
    fh.write('\n')


def write_header_ternary_grader(f):
    f.write('split,')
    f.write('white_box_name,black_box_name,grader_name,')
    f.write('white_box_mse_total,white_box_mse_easy,white_box_mse_hard,white_box_mse_very_hard,')
    f.write('black_box_mse_total,black_box_mse_easy,black_box_mse_hard,black_box_mse_very_hard,')
    f.write('final_mse_total,final_mse_easy,final_mse_hard,final_mse_very_hard,')
    f.write('deferral_rate,reject_rate')
    f.write('\n')


def save_results_ternary_grader(fh, split_idx, white_box_name, black_box_name, grader_name, results: ResultsTernary):
    fh.write(f'{split_idx},')
    fh.write(f'{white_box_name},{black_box_name},{grader_name},')
    fh.write(f'{results.white_box_mse},{results.white_box_mse_easy},{results.white_box_mse_hard},{results.white_box_mse_very_hard},')
    fh.write(f'{results.black_box_mse},{results.black_box_mse_easy},{results.black_box_mse_hard},{results.black_box_mse_very_hard},')
    fh.write(f'{results.combined_mse},{results.combined_mse_easy},{results.combined_mse_hard},{results.combined_mse_very_hard},')
    fh.write(f'{results.deferral_rate},{results.reject_rate}')
    fh.write('\n')


def save_full_results(path, X, y_truth, white_box_model, black_box_model, grader_model):
    n_features = X.shape[1]
    df = pd.DataFrame(X, index=None, columns=[f'X_{i}' for i in range(n_features)])
    df['y_truth'] = y_truth
    df['y_white'] = white_box_model.predict(X)
    df['y_black'] = black_box_model.predict(X)
    df['y_grader'] = grader_model.predict(X)
    df.to_csv(path, index=False)
