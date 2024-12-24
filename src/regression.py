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


def get_binary_grader_data(white_box_model, X_train, y_train, percentile=50):
    """
    Get data for the basic, 2-class grader
    0: Easy (Use b_easy)
    1: Hard (Use b_hard)
    Patterns are *easy* if the square error is less than or equal to the nth percentile square error
    """
    predict_train = white_box_model.predict(X_train)
    square_error = np.square(predict_train - y_train)
    hard_threshold = np.percentile(square_error, percentile)
    is_hard = (square_error > hard_threshold).astype(int)

    # This will just add one synthetic "hard" pattern...
    smote = SMOTE(k_neighbors=5, random_state=19)
    x2, y2 = smote.fit_resample(X_train, is_hard)

    return x2, y2


def get_ternary_grader_data(white_box_model, black_box_model, X_train, y_train, percentile=50):
    """
    Get data for the 3-class grader extension
    0: Easy (Use b_easy)
    1: Hard (Use b_hard)
    2: Very hard (Reject)

                        White box
                        <=50        >50
    Black-box  <=50     Easy        Hard
                >50     Very Hard   Very Hard

    Patterns are *easy* if the square error is less than or equal to the nth percentile square error

    """
    n = X_train.shape[0]
    results = np.zeros((n,), dtype=np.int64)

    white_box_predictions = white_box_model.predict(X_train)
    wb_square_error = np.square(white_box_predictions - y_train)
    wb_threshold = np.percentile(wb_square_error, percentile)

    black_box_predictions = black_box_model.predict(X_train)
    bb_square_error = np.square(black_box_predictions - y_train)
    bb_threshold = np.percentile(bb_square_error, percentile)

    hard_idx = wb_square_error > wb_threshold
    results[hard_idx] = DIFFICULTY_HARD

    vhard_idx = bb_square_error > bb_threshold
    results[vhard_idx] = DIFFICULTY_VERY_HARD

    bins = np.bincount(results)

    min_bin = np.min(bins)
    if min_bin < 2:
        logger.warning(f"Less than 2 ({min_bin}) patterns in a bin for the super grader ({bins})")
        x2, y2 = X_train.copy(), y_train.copy()
    elif bins.size < 2:
        logger.warning("Only 'easy' patterns found when training the ternary grader")
        x2, y2 = X_train.copy(), y_train.copy()
    else:
        k_neighbors = min(min_bin - 1, 5)
        smote = SMOTE(k_neighbors=k_neighbors)
        x2, y2 = smote.fit_resample(X_train, results)

    return x2, y2


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
    f.write('black_box_mse_total,black_box_mse_easy,black_box_mse_hard,black_box_mse_very_hard')
    f.write('final_mse_total,final_mse_easy,final_mse_hard,final_mse_very_hard')
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
