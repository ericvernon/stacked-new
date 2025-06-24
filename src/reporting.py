from pathlib import Path

import pandas as pd
import numpy as np

from .lib import DIFFICULTY_EASY, DIFFICULTY_HARD, DIFFICULTY_VERY_HARD


def parse_results_file(results_file: Path):
    """
    Return a dictionary with the following keys:
        - hybrid_n_correct_total: The total number of correct answers from the hybrid system
            i.e.,   glass_box_answer == y_truth AND grader_answer == EASY, PLUS
                    black_box_answer == y_truth AND grader_answer == HARD
        - hybrid_n_correct_easy: The number of correct answers from the hybrid system, limited to easy patterns
            i.e.,   glass_box_answer == y_truth AND grader_answer == EASY
        - hybrid_n_correct_hard: The number of correct answers from the hybrid system, limited to hard patterns
            i.e.,   black_box_answer == y_truth AND grader_answer == HARD
        - hybrid_n_reject: The number of patterns rejected by the hybrid system
            ie.,    grader_answer == VERY_HARD
        - glass_n_correct_total: The total number of correct answers from the glass box
            i.e.,   glass_box_answer == y_truth
        - glass_n_correct_easy: The number of correct answers from the glass box, limited to easy patterns
            i.e.,   glass_box_answer == y_truth AND grader_answer == EASY
        - glass_n_correct_hard: The number of correct answers from the glass box, limited to hard patterns
            i.e.,   glass_box_answer == y_truth AND grader_answer == HARD
        - glass_n_correct_very_hard: The number of correct answers from the glass box, limited to very hard patterns
            i.e.,   glass_box_answer == y_truth AND grader_answer == VERY_HARD
        - black_n_correct_total: Same as glass box section, but for the black box
        - black_n_correct_easy   Same as glass box section, but for the black box
        - black_n_correct_hard:  Same as glass box section, but for the black box
        - black_n_correct_very_hard:  Same as glass box section, but for the black box
        - n_total: The total number of patterns
        - n_easy: The number of patterns marked as easy by the grader
        - n_hard: The number of patterns marked as hard by the grader
        - n_very_hard: The number of patterns marked as very hard by the grader
    Some keys are not relevant to the actual results of the hybrid system (e.g. the glass box accuracy on hard patterns)
    but are included in the results report since they may provide useful insight during analysis.
    :param results_file: Filepath to parse
    :return: dict
    """
    df = pd.read_csv(results_file)

    hybrid_n_correct_total = np.count_nonzero(
        (df['y_glass'] == df['y_truth']) &
        (df['y_grader'] == DIFFICULTY_EASY)
    ) + np.count_nonzero(
        (df['y_black'] == df['y_truth']) &
        (df['y_grader'] == DIFFICULTY_HARD)
    )
    hybrid_n_correct_easy = np.count_nonzero(
        (df['y_glass'] == df['y_truth']) &
        (df['y_grader'] == DIFFICULTY_EASY)
    )
    hybrid_n_correct_hard = np.count_nonzero(
        (df['y_black'] == df['y_truth']) &
        (df['y_grader'] == DIFFICULTY_HARD)
    )
    hybrid_n_reject = np.count_nonzero(df['y_grader'] == DIFFICULTY_VERY_HARD)

    glass_n_correct_total = np.count_nonzero(df['y_glass'] == df['y_truth'])
    glass_n_correct_easy = np.count_nonzero(
        (df['y_glass'] == df['y_truth']) &
        (df['y_grader'] == DIFFICULTY_EASY)
    )
    glass_n_correct_hard = np.count_nonzero(
        (df['y_glass'] == df['y_truth']) &
        (df['y_grader'] == DIFFICULTY_HARD)
    )
    glass_n_correct_very_hard = np.count_nonzero(
        (df['y_glass'] == df['y_truth']) &
        (df['y_grader'] == DIFFICULTY_VERY_HARD)
    )

    black_n_correct_total = np.count_nonzero(df['y_black'] == df['y_truth'])
    black_n_correct_easy = np.count_nonzero(
        (df['y_black'] == df['y_truth']) &
        (df['y_grader'] == DIFFICULTY_EASY)
    )
    black_n_correct_hard = np.count_nonzero(
        (df['y_black'] == df['y_truth']) &
        (df['y_grader'] == DIFFICULTY_HARD)
    )
    black_n_correct_very_hard = np.count_nonzero(
        (df['y_black'] == df['y_truth']) &
        (df['y_grader'] == DIFFICULTY_VERY_HARD)
    )

    n_total = df.shape[0]
    n_easy = np.count_nonzero(df['y_grader'] == DIFFICULTY_EASY)
    n_hard = np.count_nonzero(df['y_grader'] == DIFFICULTY_HARD)
    n_very_hard = np.count_nonzero(df['y_grader'] == DIFFICULTY_VERY_HARD)

    return {
        'hybrid_n_correct_total': hybrid_n_correct_total,
        'hybrid_n_correct_easy': hybrid_n_correct_easy,
        'hybrid_n_correct_hard': hybrid_n_correct_hard,
        'hybrid_n_reject': hybrid_n_reject,
        'glass_n_correct_total': glass_n_correct_total,
        'glass_n_correct_easy': glass_n_correct_easy,
        'glass_n_correct_hard': glass_n_correct_hard,
        'glass_n_correct_very_hard': glass_n_correct_very_hard,
        'black_n_correct_total': black_n_correct_total,
        'black_n_correct_easy': black_n_correct_easy,
        'black_n_correct_hard': black_n_correct_hard,
        'black_n_correct_very_hard': black_n_correct_very_hard,
        'n_total': n_total,
        'n_easy': n_easy,
        'n_hard': n_hard,
        'n_very_hard': n_very_hard,
    }
