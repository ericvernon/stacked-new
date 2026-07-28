from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, confusion_matrix

import pandas as pd
import numpy as np

from .lib import DIFFICULTY_EASY, DIFFICULTY_HARD, DIFFICULTY_VERY_HARD


def load_results_from_path(results_path: Path, only_load=None):
    """
    :param results_path: Path with all the results data in it
    :param only_load: Either None, or a filter list of dataset names (ids) to only load a subset (for testing etc.)
    :return:
    """
    data_dict = {}
    for dataset in results_path.iterdir():
        if not dataset.is_dir():
            continue
        if only_load is not None and dataset.name not in only_load:
            continue
        for results_file in dataset.iterdir():
            run_info = parse_filename(results_file.name)
            algorithm_type = run_info.algorithm_type_hash()
            if algorithm_type not in data_dict:
                data_dict[algorithm_type] = {}
            if dataset.name not in data_dict[algorithm_type]:
                data_dict[algorithm_type][dataset.name] = {
                    'train': [],
                    'test': [],
                }
            data_dict[algorithm_type][dataset.name][run_info.train_or_test].append(pd.read_csv(results_file))
    return data_dict


def parse_filename(filename):
    # Filenames have this format:
    # 0-decision_tree-xgboost-dt-test-double.txt
    # Algo names can be anything. Train or test should be train/test.
    # Grader type should be 'binary' (single glass/black grader), 'ternary' (single glass/black/reject grader),
    #  or 'double' (two graders, one to decide glass/black and the other to decide accept/reject)
    bits = filename[:-4].split('-')
    return RunInfo(
        run_id=int(bits[0]),
        glass_box_algo=bits[1],
        black_box_algo=bits[2],
        grader_algo=bits[3],
        train_or_test=bits[4],
        grader_type=bits[5],
    )


@dataclass
class RunInfo:
    run_id: int
    glass_box_algo: str
    black_box_algo: str
    grader_algo: str
    train_or_test: str
    grader_type: str

    def algorithm_type_hash(self):
        return f'{self.glass_box_algo}-{self.black_box_algo}-{self.grader_algo}-{self.grader_type}'


def parse_results_dict(results_dict, grader_type):
    """
    Take in a results_dict (e.g. from load_results_from_path), and get a dictionary in the following format:
        dataset_id => {
            'train': (pandas dataframe)
            'test': (pandas dataframe)
        }
    Where each row in the pandas dataframe represents statistics about a particular run
    :param results_dict:
    :param grader_type: The type of grader to evaluate with. Choices:
        "binary"    : Assumes 0 -> Glass box, 1 -> Black box, no reject option
        "ternary"   : Assumes 0 -> GLass box, 1 -> Black box, 2 -> reject
        "double-GB" : Assumes 1st grader 0 -> Glass box, Else: 2nd grader 0 -> Black box, 1 -> reject
                        (i.e., decide glass/not-glass first, then black/reject)
        "double-AR" : Assumes 2nd grader 1 -> Reject, Else: 1st grader 0 -> Glass box, 1 -> Black box
                        (i.e., decide accept/reject first, then glass/black)
    :return:
    """
    parsed_results = {}
    for dataset_name in results_dict.keys():
        training_results = [
            parse_results_df(run_result_df, grader_type) for run_result_df in results_dict[dataset_name]['train']
        ]
        testing_results = [
            parse_results_df(run_result_df, grader_type) for run_result_df in results_dict[dataset_name]['test']
        ]
        parsed_results[dataset_name] = {
            'train': pd.DataFrame(training_results),
            'test': pd.DataFrame(testing_results),
        }

    return parsed_results


def _calc_extended_metrics(y_true, y_pred):
    """Helper to calculate macro-averaged Precision, Recall, and Specificity."""
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan

    # Macro average is used to generalize across both binary and multiclass datasets
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # Specificity (True Negative Rate) must be extracted from the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    spec_per_class = []

    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp

        if (tn + fp) > 0:
            spec_per_class.append(tn / (tn + fp))
        else:
            spec_per_class.append(0.0)

    spec = np.mean(spec_per_class) if spec_per_class else np.nan
    return prec, rec, spec


def parse_results_df(df: pd.DataFrame, grader_type: str):
    """
    Parse a *single* results dataframe, must contain columns 'y_glass', 'y_black', and 'y_truth'
    For singleton graders, should also contain a single column 'y_grader'
    For double graders, should also contain two columns, 'y_grader' and 'y_grader2'
    :param df: The dataframe to parse
    :param grader_type: The type of grader to evaluate with
    :return: A Python dictionary with parsed statistics about that run
    """
    # Assuming these are defined elsewhere in your file:
    DIFFICULTY_EASY = 0
    DIFFICULTY_HARD = 1
    DIFFICULTY_VERY_HARD = 2

    if grader_type == 'binary' or grader_type == 'ternary':
        y_final_grade = df['y_grader'].copy()
    elif grader_type == 'double-GB':
        hard_idx = (df['y_grader'] == 1) & (df['y_grader2'] == 0)
        vhard_idx = (df['y_grader'] == 1) & (df['y_grader2'] == 1)
        y_final_grade = np.zeros_like(df['y_grader'])
        y_final_grade[hard_idx] = DIFFICULTY_HARD
        y_final_grade[vhard_idx] = DIFFICULTY_VERY_HARD
    elif grader_type == 'double-AR':
        hard_idx = (df['y_grader'] == 1) & (df['y_grader2'] == 0)
        vhard_idx = (df['y_grader2'] == 1)
        y_final_grade = np.zeros_like(df['y_grader'])
        y_final_grade[hard_idx] = DIFFICULTY_HARD
        y_final_grade[vhard_idx] = DIFFICULTY_VERY_HARD
    else:
        raise ValueError(f'Unknown grader type {grader_type}')
    df['y_difficulty'] = y_final_grade

    # --- ACCURACY CALCULATIONS ---
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

    # --- EXTENDED METRICS (Precision, Recall, Specificity) ---
    glass_prec, glass_rec, glass_spec = _calc_extended_metrics(df['y_truth'], df['y_glass'])
    black_prec, black_rec, black_spec = _calc_extended_metrics(df['y_truth'], df['y_black'])

    # For Hybrid, we must filter out the Rejected patterns (VERY_HARD)
    valid_hybrid_mask = (df['y_difficulty'] != DIFFICULTY_VERY_HARD)
    y_truth_hybrid = df['y_truth'][valid_hybrid_mask]

    # Reconstruct the Hybrid's actual predictions based on the difficulty flag
    y_pred_hybrid = np.where(
        df['y_difficulty'][valid_hybrid_mask] == DIFFICULTY_EASY,
        df['y_glass'][valid_hybrid_mask],
        df['y_black'][valid_hybrid_mask]
    )
    hybrid_prec, hybrid_rec, hybrid_spec = _calc_extended_metrics(y_truth_hybrid, y_pred_hybrid)

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
        'hybrid_accuracy_all': hybrid_n_correct_total / (
                n_total - hybrid_n_reject) if n_total > hybrid_n_reject else np.nan,
        'hybrid_accuracy_easy': (hybrid_n_correct_easy / n_easy) if n_easy > 0 else np.nan,
        'hybrid_accuracy_hard': (hybrid_n_correct_hard / n_hard) if n_hard > 0 else np.nan,

        # --- NEW METRICS ADDED HERE ---
        'glass_precision': glass_prec,
        'glass_recall': glass_rec,
        'glass_specificity': glass_spec,
        'black_precision': black_prec,
        'black_recall': black_rec,
        'black_specificity': black_spec,
        'hybrid_precision': hybrid_prec,
        'hybrid_recall': hybrid_rec,
        'hybrid_specificity': hybrid_spec,
        # ------------------------------

        'glass_n_correct_total': glass_n_correct_total,
        'glass_n_correct_easy': glass_n_correct_easy,
        'glass_n_correct_hard': glass_n_correct_hard,
        'glass_n_correct_very_hard': glass_n_correct_very_hard,
        'glass_accuracy_all': glass_n_correct_total / n_total,
        'glass_accuracy_easy': (glass_n_correct_easy / n_easy) if n_easy > 0 else np.nan,
        'glass_accuracy_hard': (glass_n_correct_hard / n_hard) if n_hard > 0 else np.nan,
        'glass_accuracy_very_hard': (glass_n_correct_very_hard / n_very_hard) if n_very_hard > 0 else np.nan,
        'glass_accuracy_non_easy': ((glass_n_correct_hard + glass_n_correct_very_hard) / (n_hard + n_very_hard)) if (
                                                                                                                                n_hard + n_very_hard) > 0 else np.nan,
        'black_n_correct_total': black_n_correct_total,
        'black_n_correct_easy': black_n_correct_easy,
        'black_n_correct_hard': black_n_correct_hard,
        'black_n_correct_very_hard': black_n_correct_very_hard,
        'black_accuracy_all': black_n_correct_total / n_total,
        'black_accuracy_easy': (black_n_correct_easy / n_easy) if n_easy > 0 else np.nan,
        'black_accuracy_hard': (black_n_correct_hard / n_hard) if n_hard > 0 else np.nan,
        'black_accuracy_very_hard': (black_n_correct_very_hard / n_very_hard) if n_very_hard > 0 else np.nan,
        'black_accuracy_non_easy': ((black_n_correct_hard + black_n_correct_very_hard) / (n_hard + n_very_hard)) if (
                                                                                                                                n_hard + n_very_hard) > 0 else np.nan,
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


def difficulty_band_summary_latex(parsed_results):
    sb_train, sb_test = '', ''
    for dataset_name, results in parsed_results.items():
        sb_train += band_summary_helper(dataset_name, results['train'])
        sb_test += band_summary_helper(dataset_name, results['test'])

    return sb_train, sb_test


def band_summary_helper(dataset_name, df):
    return (f'{dataset_names[dataset_name]} & '
            f'{100 * df['hybrid_accuracy_all'].mean():.2f} & '
            f'$\\pm$ {100 * df['hybrid_accuracy_all'].std():.2f} & '
            f'{100 * df['glass_accuracy_all'].mean():.2f} & '
            f'$\\pm$ {100 * df['glass_accuracy_all'].std():.2f} & '
            f'{100 * df['black_accuracy_all'].mean():.2f} & '
            f'$\\pm$ {100 * df['black_accuracy_all'].std():.2f} & '
            f'{100 * df['glass_accuracy_easy'].mean():.2f} & '
            f'$\\pm$ {100 * df['glass_accuracy_easy'].std():.2f} & '
            f'{100 * df['black_accuracy_easy'].mean():.2f} & '
            f'$\\pm$ {100 * df['black_accuracy_easy'].std():.2f} & '
            f'{100 * df['glass_accuracy_hard'].mean():.2f} & '
            f'$\\pm$ {100 * df['glass_accuracy_hard'].std():.2f} & '
            f'{100 * df['black_accuracy_hard'].mean():.2f} & '
            f'$\\pm$ {100 * df['black_accuracy_hard'].std():.2f} & '
            f'{100 * df['glass_accuracy_very_hard'].mean():.2f} & '
            f'$\\pm$ {100 * df['glass_accuracy_very_hard'].std():.2f} & '
            f'{100 * df['black_accuracy_very_hard'].mean():.2f} & '
            f'$\\pm$ {100 * df['black_accuracy_very_hard'].std():.2f} \\\\ \n'
            )


def results_overview_latex(parsed_results_with_reject, parsed_results_no_reject):
    sb_train, sb_test = '', ''
    for dataset_name in parsed_results_with_reject.keys():
        results_reject = parsed_results_with_reject[dataset_name]
        results_no_reject = parsed_results_no_reject[dataset_name]
        sb_train += results_overview_helper(dataset_name, results_reject['train'], results_no_reject['train'])
        sb_test += results_overview_helper(dataset_name, results_reject['test'], results_no_reject['test'])
    return sb_train, sb_test


def results_overview_helper(dataset_name, df_with_reject, df_no_reject):
    # 1. Prepare the data for the 4 methods being compared
    # Structure: (Mean, Std)
    methods = [
        (df_with_reject['glass_accuracy_all'].mean(), df_with_reject['glass_accuracy_all'].std()),
        (df_with_reject['black_accuracy_all'].mean(), df_with_reject['black_accuracy_all'].std()),
        (df_no_reject['hybrid_accuracy_all'].mean(), df_no_reject['hybrid_accuracy_all'].std()),
        (df_with_reject['hybrid_accuracy_all'].mean(), df_with_reject['hybrid_accuracy_all'].std())
    ]

    # 2. Find the highest mean accuracy among the 4 methods
    max_accuracy = max(m[0] for m in methods)

    # 3. Helper to format cells (Bolds if the mean matches max_accuracy)
    def fmt_acc(mean, std):
        m_str = f"{100 * mean:.2f}"
        s_str = f"$\\pm$ {100 * std:.2f}"
        # If this is the winner, bold both the Mean and the Std
        if mean == max_accuracy:
            return f"\\textbf{{{m_str}}} & \\textbf{{{s_str}}}"
        return f"{m_str} & {s_str}"

    # 4. Standard formatting for the non-compared metrics (Usage & Reject Rate)
    def fmt_stat(col_name):
        mean = df_with_reject[col_name].mean()
        std = df_with_reject[col_name].std()
        return f"{100 * mean:.2f} & $\\pm$ {100 * std:.2f}"

    # Construct the row
    return (
        f"{dataset_names[dataset_name]} & "
        f"{fmt_acc(*methods[0])} & "  # DT Accuracy
        f"{fmt_acc(*methods[1])} & "  # XGB Accuracy
        f"{fmt_acc(*methods[2])} & "  # Hybrid (No Reject)
        f"{fmt_acc(*methods[3])} & "  # Hybrid (Reject)
        f"{fmt_stat('hybrid_glass_usage')} & "
        f"{fmt_stat('hybrid_black_usage')} & "
        f"{fmt_stat('hybrid_reject_rate')} "
        "\\\\ \n"
    )


def results_extended_latex(parsed_results_with_reject, parsed_results_no_reject):
    sb_train, sb_test = '', ''
    for dataset_name in parsed_results_with_reject.keys():
        results_reject = parsed_results_with_reject[dataset_name]
        results_no_reject = parsed_results_no_reject[dataset_name]
        sb_train += extended_results_helper(dataset_name, results_reject['train'], results_no_reject['train'])
        sb_test += extended_results_helper(dataset_name, results_reject['test'], results_no_reject['test'])
    return sb_train, sb_test


def extended_results_helper(dataset_name, df_with_reject, df_no_reject):
    # 1. Prepare the data for the 4 methods being compared
    # Structure: (Mean, Std)
    methods = [
        (
            df_with_reject['glass_precision'].mean(), df_with_reject['glass_precision'].std(),
            df_with_reject['glass_recall'].mean(), df_with_reject['glass_recall'].std(),
            df_with_reject['glass_specificity'].mean(), df_with_reject['glass_specificity'].std(),
        ),
        (
            df_with_reject['black_precision'].mean(), df_with_reject['black_precision'].std(),
            df_with_reject['black_recall'].mean(), df_with_reject['black_recall'].std(),
            df_with_reject['black_specificity'].mean(), df_with_reject['black_specificity'].std(),
        ),
        (
            df_no_reject['hybrid_precision'].mean(), df_no_reject['hybrid_precision'].std(),
            df_no_reject['hybrid_recall'].mean(), df_no_reject['hybrid_recall'].std(),
            df_no_reject['hybrid_specificity'].mean(), df_no_reject['hybrid_specificity'].std(),
        ),
        (
            df_with_reject['hybrid_precision'].mean(), df_with_reject['hybrid_precision'].std(),
            df_with_reject['hybrid_recall'].mean(), df_with_reject['hybrid_recall'].std(),
            df_with_reject['hybrid_specificity'].mean(), df_with_reject['hybrid_specificity'].std(),
        ),
    ]

    max_prec = max(m[0] for m in methods)
    max_rec = max(m[2] for m in methods)
    max_spec = max(m[4] for m in methods)

    # 3. Helper to format a single value, bolding it if it matches the maximum
    def fmt_val(val, max_val):
        formatted_val = f"{100 * val:.2f}"  # Using 1 decimal place to save horizontal space!

        # Use np.isclose to avoid strict floating-point equality bugs
        if np.isclose(val, max_val, atol=1e-5):
            return f"\\textbf{{{formatted_val}}}"
        return formatted_val

    # 4. Helper to format a complete method's metrics
    def fmt_method(m):
        return f"{fmt_val(m[0], max_prec)} & {fmt_val(m[2], max_rec)} & {fmt_val(m[4], max_spec)}"

    # 5. Construct the final LaTeX row
    return (
        f"{dataset_names[dataset_name]} & "
        f"{fmt_method(methods[0])} & "  # DT
        f"{fmt_method(methods[1])} & "  # XGB
        f"{fmt_method(methods[2])} & "  # Hybrid (No Reject)
        f"{fmt_method(methods[3])} \\\\ \n"  # Hybrid (Reject)
    )



def simple_accuracy_table(parsed_results_with_reject, parsed_results_no_reject):
    header = 'dataset,gb_accuracy,bb_accuracy,hybrid_accuracy_no_reject,hybrid_accuracy_w_reject\n'
    sb_train, sb_test = header, header
    for dataset_name in parsed_results_with_reject.keys():
        results_reject = parsed_results_with_reject[dataset_name]
        results_no_reject = parsed_results_no_reject[dataset_name]
        sb_train += simple_accuracy_helper(dataset_name, results_reject['train'], results_no_reject['train'])
        sb_test += simple_accuracy_helper(dataset_name, results_reject['test'], results_no_reject['test'])
    return sb_train, sb_test

def simple_accuracy_helper(dataset_name, df_with_reject, df_no_reject):
    # dataset name | DT Accuracy | XGB Accuracy | Hybrid Accuracy (No Reject) | Hybrid Accuracy (Reject) |
    return (
        f'{dataset_names[dataset_name]},'
        f'{df_with_reject['glass_accuracy_all'].mean():.8f},'
        f'{df_with_reject['black_accuracy_all'].mean():.8f},'
        f'{df_no_reject['hybrid_accuracy_all'].mean():.8f},'
        f'{df_with_reject['hybrid_accuracy_all'].mean():.8f}'
        f'\n'
    )

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


def bulk_text_report(parsed_results):
    sb = ''
    for dataset_name in parsed_results.keys():
        sb += f'DATASET ID: {dataset_name} ({dataset_names[dataset_name]})\n'
        sb += '--TRAIN--\n'
        sb += results_df_to_text(parsed_results[dataset_name]['train'])
        sb += '--TEST--\n'
        sb += results_df_to_text(parsed_results[dataset_name]['test'])
        sb += '\n'

    return sb


def csv_summary_report(parsed_results, prefix):
    sb = ''
    for dataset_name, results in parsed_results.items():
        sb += csv_summary_from_df(results['train'], results['test'], dataset_name, prefix)
    return sb


CSV_SUMMARY_HEADER = ('dataset,algorithm,'
                      'hybrid_accuracy_train,hybrid_glass_train,hybrid_black_train,hybrid_reject_train,hybrid_kappa_train,'
                      'hybrid_accuracy_test,hybrid_glass_test,hybrid_black_test,hybrid_reject_test,hybrid_kappa_test')


def csv_summary_from_df(df_train, df_test, dataset_name, prefix):
    return f'{dataset_name},'\
           f'{prefix},'\
           f'{df_train['hybrid_accuracy_all'].mean():.8f},'\
           f'{df_train['hybrid_glass_usage'].mean():.8f},'\
           f'{df_train['hybrid_black_usage'].mean():.8f},'\
           f'{df_train['hybrid_reject_rate'].mean():.8f},'\
           f'{df_train['grader_kappa'].mean():.8f},'\
           f'{df_test['hybrid_accuracy_all'].mean():.8f},' \
           f'{df_test['hybrid_glass_usage'].mean():.8f},' \
           f'{df_test['hybrid_black_usage'].mean():.8f},' \
           f'{df_test['hybrid_reject_rate'].mean():.8f},'\
           f'{df_test['grader_kappa'].mean():.8f}\n'


dataset_names = {
    '17': 'breast-w',
    '19': 'car',
    '43': 'haberman',
    '52': 'ionosphere',
    '59': 'letter',
    '78': 'page-block',
    '94': 'spambase',
    '96': 'sp-hrt',
    '151': 'sonar',
    '159': 'magic',
    '174': 'parkinsons',
    '176': 'blood',
    '212': 'vertebral',
    '267': 'banknote',
    '329': 'retinopath',
    '372': 'htru2',
    '451': 'breast-c',
    '519': 'heart-f',
    '537': 'cervical',
    '545': 'rice',
    '563': 'iranian-ch',
    '572': 'taiwan-bnk',
    '602': 'dry-bean',
    '722': 'droid',
#    '732': 'darwin',
#    '759': 'glioma',
    '827': 'sepsis',
    '850': 'raisin',
    '863': 'mat-health',
    '887': 'nhanes',
    '890': 'aids',
    '891': 'cdc-diabet',
}
