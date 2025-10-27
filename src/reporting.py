from pathlib import Path
from sklearn.metrics import cohen_kappa_score

import pandas as pd
import numpy as np

from .lib import DIFFICULTY_EASY, DIFFICULTY_HARD, DIFFICULTY_VERY_HARD


def parse_results_file(results_file: Path, grader_type: str):
    """
    Input: a *raw* results file, which just has the glass box answer, black box answer, ground truth, etc.
    Output: a dictionary with the following keys:
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


    df['y_difficulty'] = df['y_grader']
    if grader_type == 'double':
        # Glass/Black Priority
        reject_idx = ((df['y_grader2'] == 1) & (df['y_grader'] != 0))
        # Accept/Reject Priority
        #reject_idx = df['y_grader2'] == 1
        df.loc[reject_idx, 'y_difficulty'] = DIFFICULTY_VERY_HARD

    return parse_results_df(df, grader_type)


def parse_results_df(df: pd.DataFrame, grader_type: str):
    """
    Input: A dataframe representing the *raw* results (glass box prediction, black box prediction, ground truth, etc.) for one single run
    Output: A parsed dictionary summarizing those results
    """
    hybrid_n_correct_easy = np.count_nonzero(
        (df['y_glass'] == df['y_truth']) &
        (df['y_difficulty'] == DIFFICULTY_EASY)
    )
    hybrid_n_correct_hard = np.count_nonzero(
        (df['y_black'] == df['y_truth']) &
        (df['y_difficulty'] == DIFFICULTY_HARD)
    )
    hybrid_n_correct_total = hybrid_n_correct_easy + hybrid_n_correct_hard
    hybrid_n_reject = np.count_nonzero(df['y_difficulty'] == DIFFICULTY_VERY_HARD)

    glass_n_correct_total = np.count_nonzero(df['y_glass'] == df['y_truth'])
    glass_n_correct_easy = np.count_nonzero(
        (df['y_glass'] == df['y_truth']) &
        (df['y_difficulty'] == DIFFICULTY_EASY)
    )
    glass_n_correct_hard = np.count_nonzero(
        (df['y_glass'] == df['y_truth']) &
        (df['y_difficulty'] == DIFFICULTY_HARD)
    )
    glass_n_correct_very_hard = np.count_nonzero(
        (df['y_glass'] == df['y_truth']) &
        (df['y_difficulty'] == DIFFICULTY_VERY_HARD)
    )

    black_n_correct_total = np.count_nonzero(df['y_black'] == df['y_truth'])
    black_n_correct_easy = np.count_nonzero(
        (df['y_black'] == df['y_truth']) &
        (df['y_difficulty'] == DIFFICULTY_EASY)
    )
    black_n_correct_hard = np.count_nonzero(
        (df['y_black'] == df['y_truth']) &
        (df['y_difficulty'] == DIFFICULTY_HARD)
    )
    black_n_correct_very_hard = np.count_nonzero(
        (df['y_black'] == df['y_truth']) &
        (df['y_difficulty'] == DIFFICULTY_VERY_HARD)
    )

    # True-difficulty selectors
    actually_easy_selector = (df['y_glass'] == df['y_truth'])
    if grader_type == 'binary':
        actually_hard_selector = (df['y_glass'] != df['y_truth'])
        actually_vhard_selector = np.zeros_like(actually_hard_selector)
    else:
        actually_hard_selector = ((df['y_glass'] != df['y_truth']) & (df['y_black'] == df['y_truth']))
        actually_vhard_selector = ((df['y_glass'] != df['y_truth']) & (df['y_black'] != df['y_truth']))

    # Row totals
    n_actually_easy = np.count_nonzero(actually_easy_selector)
    n_actually_hard = np.count_nonzero(actually_hard_selector)
    n_actually_vhard = np.count_nonzero(actually_vhard_selector)

    # Row: Actually Easy
    n_easy_as_easy = np.count_nonzero(actually_easy_selector & (df['y_difficulty'] == DIFFICULTY_EASY))
    n_easy_as_hard = np.count_nonzero(actually_easy_selector & (df['y_difficulty'] == DIFFICULTY_HARD))
    n_easy_as_vhard = np.count_nonzero(actually_easy_selector & (df['y_difficulty'] == DIFFICULTY_VERY_HARD))

    # Row: Actually Hard
    n_hard_as_easy = np.count_nonzero(actually_hard_selector & (df['y_difficulty'] == DIFFICULTY_EASY))
    n_hard_as_hard = np.count_nonzero(actually_hard_selector & (df['y_difficulty'] == DIFFICULTY_HARD))
    n_hard_as_vhard = np.count_nonzero(actually_hard_selector & (df['y_difficulty'] == DIFFICULTY_VERY_HARD))

    # Row: Actually Very Hard
    n_vhard_as_easy = np.count_nonzero(actually_vhard_selector & (df['y_difficulty'] == DIFFICULTY_EASY))
    n_vhard_as_hard = np.count_nonzero(actually_vhard_selector & (df['y_difficulty'] == DIFFICULTY_HARD))
    n_vhard_as_vhard = np.count_nonzero(actually_vhard_selector & (df['y_difficulty'] == DIFFICULTY_VERY_HARD))

    n_total = df.shape[0]
    n_easy = np.count_nonzero(df['y_difficulty'] == DIFFICULTY_EASY)
    n_hard = np.count_nonzero(df['y_difficulty'] == DIFFICULTY_HARD)
    n_very_hard = np.count_nonzero(df['y_difficulty'] == DIFFICULTY_VERY_HARD)

    actual_difficulty = np.full(shape=(df.shape[0]), fill_value=DIFFICULTY_EASY)
    actual_difficulty[actually_hard_selector] = DIFFICULTY_HARD
    actual_difficulty[actually_vhard_selector] = DIFFICULTY_VERY_HARD
    if len(np.unique(df['y_difficulty'])) > 1 and len(np.unique(actual_difficulty)) > 1:
        kappa_score = cohen_kappa_score(df['y_difficulty'], actual_difficulty, labels=[0, 1, 2], weights='quadratic')
    else:
        kappa_score = np.nan

    return {
        'hybrid_n_correct_total': hybrid_n_correct_total,
        'hybrid_n_correct_easy': hybrid_n_correct_easy,
        'hybrid_n_correct_hard': hybrid_n_correct_hard,
        'hybrid_n_reject': hybrid_n_reject,
        'hybrid_accuracy_all': hybrid_n_correct_total / (n_total - hybrid_n_reject) if n_total > hybrid_n_reject else np.nan,
        'hybrid_accuracy_easy': (hybrid_n_correct_easy / n_easy) if n_easy > 0 else np.nan,
        'hybrid_accuracy_hard': (hybrid_n_correct_hard / n_hard) if n_hard > 0 else np.nan,
        'glass_n_correct_total': glass_n_correct_total,
        'glass_n_correct_easy': glass_n_correct_easy,
        'glass_n_correct_hard': glass_n_correct_hard,
        'glass_n_correct_very_hard': glass_n_correct_very_hard,
        'glass_accuracy_all': glass_n_correct_total / n_total,
        'glass_accuracy_easy': (glass_n_correct_easy / n_easy) if n_easy > 0 else np.nan,
        'glass_accuracy_hard': (glass_n_correct_hard / n_hard) if n_hard > 0 else np.nan,
        'glass_accuracy_very_hard': (glass_n_correct_very_hard / n_very_hard) if n_very_hard > 0 else np.nan,
        'glass_accuracy_non_easy': ((glass_n_correct_hard + glass_n_correct_very_hard) / (n_hard + n_very_hard)) if (n_hard + n_very_hard) > 0 else np.nan,
        'black_n_correct_total': black_n_correct_total,
        'black_n_correct_easy': black_n_correct_easy,
        'black_n_correct_hard': black_n_correct_hard,
        'black_n_correct_very_hard': black_n_correct_very_hard,
        'black_accuracy_all': black_n_correct_total / n_total,
        'black_accuracy_easy': (black_n_correct_easy / n_easy)  if n_easy > 0 else np.nan,
        'black_accuracy_hard': (black_n_correct_hard / n_hard) if n_hard > 0 else np.nan,
        'black_accuracy_very_hard': (black_n_correct_very_hard / n_very_hard) if n_very_hard > 0 else np.nan,
        'black_accuracy_non_easy': ((black_n_correct_hard + black_n_correct_very_hard) / (n_hard + n_very_hard)) if (n_hard + n_very_hard) > 0 else np.nan,
        'hybrid_glass_usage': n_easy / n_total,
        'hybrid_black_usage': n_hard / n_total,
        'hybrid_reject_rate': n_very_hard / n_total,
        'glass_correct_easy_rate': (glass_n_correct_easy / glass_n_correct_total),
        'glass_correct_hard_rate': (glass_n_correct_hard / glass_n_correct_total),
        'glass_correct_reject_rate': (glass_n_correct_very_hard / glass_n_correct_total),
        'black_correct_easy_rate': (black_n_correct_easy / black_n_correct_total),
        'black_correct_hard_rate': (black_n_correct_hard / black_n_correct_total),
        'black_correct_reject_rate': (black_n_correct_very_hard / black_n_correct_total),
        'n_total': n_total,
        'n_easy': n_easy,
        'n_hard': n_hard,
        'n_very_hard': n_very_hard,
        'pct_easy_as_easy': n_easy_as_easy / n_actually_easy if n_actually_easy > 0 else np.nan,
        'pct_easy_as_hard': n_easy_as_hard / n_actually_easy if n_actually_easy > 0 else np.nan,
        'pct_easy_as_vhard': n_easy_as_vhard / n_actually_easy if n_actually_easy > 0 else np.nan,
        'pct_hard_as_easy': n_hard_as_easy / n_actually_hard if n_actually_hard > 0 else np.nan,
        'pct_hard_as_hard': n_hard_as_hard / n_actually_hard if n_actually_hard > 0 else np.nan,
        'pct_hard_as_vhard': n_hard_as_vhard / n_actually_hard if n_actually_hard > 0 else np.nan,
        'pct_vhard_as_easy': n_vhard_as_easy / n_actually_vhard if n_actually_vhard > 0 else np.nan,
        'pct_vhard_as_hard': n_vhard_as_hard / n_actually_vhard if n_actually_vhard > 0 else np.nan,
        'pct_vhard_as_vhard': n_vhard_as_vhard / n_actually_vhard if n_actually_vhard > 0 else np.nan,
        'grader_kappa': kappa_score,
    }


def results_df_to_text(results_df):
    sb = ''
    pct_easy = results_df['hybrid_glass_usage'].mean()
    pct_hard = results_df['hybrid_black_usage'].mean()
    pct_very_hard = results_df['hybrid_reject_rate'].mean()
    sb += (
        '  Pattern Allocation\n'
        '|| Easy       || Hard       || Very Hard  ||\n'
        '||----------------------------------------||\n'
        f'||{100 * pct_easy:6.2f}%     ||{100 * pct_hard:6.2f}%     ||{100 * pct_very_hard:6.2f}%     ||\n'
        '||----------------------------------------||\n'
        '\n'
    )

    hybrid_overall = results_df['hybrid_accuracy_all'].mean()
    hybrid_easy = results_df['hybrid_accuracy_easy'].mean()
    hybrid_hard = results_df['hybrid_accuracy_hard'].mean()

    glass_overall = results_df['glass_accuracy_all'].mean()
    glass_easy = results_df['glass_accuracy_easy'].mean()
    glass_hard = results_df['glass_accuracy_hard'].mean()
    glass_very_hard = results_df['glass_accuracy_very_hard'].mean()
    glass_non_easy = results_df['glass_accuracy_non_easy'].mean()

    black_overall = results_df['black_accuracy_all'].mean()
    black_easy = results_df['black_accuracy_easy'].mean()
    black_hard = results_df['black_accuracy_hard'].mean()
    black_very_hard = results_df['black_accuracy_very_hard'].mean()
    black_non_easy = results_df['black_accuracy_non_easy'].mean()

    pct_easy_as_easy = results_df['pct_easy_as_easy'].mean()
    pct_easy_as_hard = results_df['pct_easy_as_hard'].mean()
    pct_easy_as_vhard = results_df['pct_easy_as_vhard'].mean()

    pct_hard_as_easy = results_df['pct_hard_as_easy'].mean()
    pct_hard_as_hard = results_df['pct_hard_as_hard'].mean()
    pct_hard_as_vhard = results_df['pct_hard_as_vhard'].mean()

    pct_vhard_as_easy = results_df['pct_vhard_as_easy'].mean()
    pct_vhard_as_hard = results_df['pct_vhard_as_hard'].mean()
    pct_vhard_as_vhard = results_df['pct_vhard_as_vhard'].mean()

    grader_kappa = results_df['grader_kappa'].mean()

    sb += (
        '  Accuracy\n'
        '||            || Overall    || Easy       || Hard       || Very Hard  || (Hard + V. Hard) ||\n'
        '||------------||------------||------------||------------||------------||------------------||\n'
        f'|| Hybrid     ||{100 * hybrid_overall:6.2f}%     ||{100 * hybrid_easy:6.2f}%     ||'
        f'{100 * hybrid_hard:6.2f}%     || (N/A)      || (N/A)            ||\n'
        f'|| Glass Box  ||{100 * glass_overall:6.2f}%     ||{100 * glass_easy:6.2f}%     ||'
        f'{100 * glass_hard:6.2f}%     ||{100 * glass_very_hard:6.2f}%     ||{100 * glass_non_easy:6.2f}%           ||\n'
        f'|| Black Box  ||{100 * black_overall:6.2f}%     ||{100 * black_easy:6.2f}%     ||'
        f'{100 * black_hard:6.2f}%     ||{100 * black_very_hard:6.2f}%     ||{100 * black_non_easy:6.2f}%           ||\n'
        '||------------||------------||------------||------------||------------||------------------||\n'
        '\n'
    )
    sb += (
        f'   Grader Consistency         Percent allocated as....      Kappa: {grader_kappa:.4f} \n'
        f'|| Actually Difficulty     || Easy        || Hard        || Very Hard  ||\n'
        f'||-------------------------||-------------||-------------||------------||\n'
        f'|| Easy (Glass Correct)    || {100 * pct_easy_as_easy:6.2f}%     || {100 * pct_easy_as_hard:6.2f}%     || {100 * pct_easy_as_vhard:6.2f}%    ||\n'
        f'|| Hard (Black Correct)    || {100 * pct_hard_as_easy:6.2f}%     || {100 * pct_hard_as_hard:6.2f}%     || {100 * pct_hard_as_vhard:6.2f}%    ||\n'
        f'|| V. Hard (Should Reject) || {100 * pct_vhard_as_easy:6.2f}%     || {100 * pct_vhard_as_hard:6.2f}%     || {100 * pct_vhard_as_vhard:6.2f}%    ||\n'
        f'||-------------------------||-------------||-------------||------------||\n'
        '\n'
    )



    return sb


def bulk_text_report(reports_path, results):
    for algorithm_type, algorithm_results in results.items():
        report = ''
        for dataset_name in algorithm_results.keys():
            df_train = pd.DataFrame.from_dict(algorithm_results[dataset_name]['train']).set_index('run_id')
            df_test = pd.DataFrame.from_dict(algorithm_results[dataset_name]['test']).set_index('run_id')
            report += f'DATASET ID: {dataset_name} ({dataset_names[dataset_name]})\n'
            report += '--TRAIN--\n'
            report += results_df_to_text(df_train)
            report += '--TEST--\n'
            report += results_df_to_text(df_test)
            report += '\n'
        report_path = f'{reports_path / algorithm_type}.txt'
        with open(report_path, 'w', encoding='UTF-8') as f:
            f.write(report)


def csv_summary_report(path, results, algo_prefix=''):
    with open(path, 'w', encoding='UTF-8') as f:
        f.write("dataset,algorithm,train_acc_mean,train_reject_mean,train_grader_kappa_mean,test_acc_mean,test_reject_mean,test_grader_kappa_mean\n")
        for algorithm_type, algorithm_results in results.items():
            for dataset_name in algorithm_results.keys():
                df_train = pd.DataFrame.from_dict(algorithm_results[dataset_name]['train']).set_index('run_id')
                df_test = pd.DataFrame.from_dict(algorithm_results[dataset_name]['test']).set_index('run_id')
                f.write(
                    f'{dataset_names[dataset_name]},'
                    f'{algo_prefix}{algorithm_type},'
                    f'{df_train['hybrid_accuracy_all'].mean():.8f},'
                    f'{df_train['hybrid_reject_rate'].mean():.8f},'
                    f'{df_train['grader_kappa'].mean():.8f},'
                    f'{df_test['hybrid_accuracy_all'].mean():.8f},'
                    f'{df_test['hybrid_reject_rate'].mean():.8f},'
                    f'{df_test['grader_kappa'].mean():.8f}\n'
                )

#
# def bulk_latex_report(train_or_test):
#     report = ''
#     reports_path = reports_root / experiment_name
#     report_path = reports_path / f'latex_for_paper_{train_or_test}.txt'
#     results = load_results()
#     bg_key = 'decision_tree-xgboost-decision_tree-binary'
#     tg_key = 'decision_tree-xgboost-decision_tree-ternary'
#     assert(results[bg_key].keys() == results[tg_key].keys())
#     dataset_ids = results[bg_key].keys()
#     for dataset_id in dataset_ids:
#         if dataset_id in dataset_names:
#             dset_name = dataset_names[dataset_id]
#         else:
#             dset_name = dataset_id
#         report += f'{dset_name} & '
#         binary_grader = latex_df_from_dict(results[bg_key][dataset_id][train_or_test])
#         ternary_grader = latex_df_from_dict(results[tg_key][dataset_id][train_or_test])
#
#         print(ternary_grader['hybrid_accuracy'].mean())
#
#         # Latex style:
#         # brst-w & 96.56 & ± 1.00 & 99.86 & ± 0.23 & 99.74 & ± 0.30 & 23.32 & ± 8.39 & 23.32 & ± 8.39 & 23.32 & ± 8.39 & 23.32 & ± 8.39\\ \hline
#         report += f'{(100 * binary_grader["glass_accuracy"].mean()):.2f} & '
#         report += f'± {(100 * binary_grader["glass_accuracy"].std()):.2f} & '
#
#         report += f'{(100 * binary_grader["black_accuracy"].mean()):.2f} & '
#         report += f'± {(100 * binary_grader["black_accuracy"].std()):.2f} & '
#
#         report += f'{(100 * binary_grader["hybrid_accuracy"].mean()):.2f} & '
#         report += f'± {(100 * binary_grader["hybrid_accuracy"].std()):.2f} & '
#
#         report += f'{(100 * ternary_grader["hybrid_accuracy"].mean()):.2f} & '
#         report += f'± {(100 * ternary_grader["hybrid_accuracy"].std()):.2f} & '
#
#         report += f'{(100 * ternary_grader["glass_usage"].mean()):.2f} & '
#         report += f'± {(100 * ternary_grader["glass_usage"].std()):.2f} & '
#
#         report += f'{(100 * ternary_grader["black_usage"].mean()):.2f} & '
#         report += f'± {(100 * ternary_grader["black_usage"].std()):.2f} & '
#
#         report += f'{(100 * ternary_grader["reject_rate"].mean()):.2f} & '
#         report += f'± {(100 * ternary_grader["reject_rate"].std()):.2f}\\\\ \\hline'
#         report += '\n'
#     print(dataset_ids)
#     with open(report_path, 'w', encoding='UTF-8') as f:
#         f.write(report)
#     return
#
#
# def latex_df_from_dict(results_dict):
#     df = pd.DataFrame.from_dict(results_dict).set_index('run_id')
#     df['glass_accuracy'] = df['glass_n_correct_total'] / df['n_total']
#     df['black_accuracy'] = df['black_n_correct_total'] / df['n_total']
#     df['hybrid_accuracy'] = df['hybrid_n_correct_total'] / (df['n_total'] - df['hybrid_n_reject'])
#     df['glass_usage'] = df['n_easy'] / df['n_total']
#     df['black_usage'] = df['n_hard'] / df['n_total']
#     df['reject_rate'] = df['n_very_hard'] / df['n_total']
#     return df


dataset_names = {
    '17': 'brst-w',
    '19': 'car',
    '43': 'haber',
    '52': 'iono',
    '59': 'letter',
    '78': 'page',
    '94': 'spam',
    '96': 'sp-hrt',
    '151': 'bench',
    '159': 'gamma',
    '174': 'parkns',
    '176': 'blood',
    '212': 'vertebr',
    '267': 'bank',
    '329': 'diabet',
    '372': 'htru',
    '451': 'brst-c',
    '519': 'heart-f',
    '537': 'cerv-c',
    '545': 'rice',
    '563': 'churn',
    '572': 'taiwan',
    '602': 'drybean',
    '722': 'droid',
    '732': 'darwin',
    '759': 'glioma',
    '827': 'sepsis',
    '850': 'raisin',
    '863': 'matern',
    '887': 'survey',
    '890': 'aids',
    '891': 'cdc-d',
}
