import pandas as pd

from dataclasses import dataclass
from pathlib import Path

from src.reporting import parse_results_file, results_dict_to_text

reports_root = Path('./output/reports')
results_root = Path('./output/results')

results_keys = [
    'glass_n_correct_easy',
    'glass_n_correct_hard',
    'glass_n_correct_very_hard',
    'glass_n_correct_total',
    'black_n_correct_easy',
    'black_n_correct_hard',
    'black_n_correct_very_hard',
    'black_n_correct_total',
    'hybrid_n_correct_easy',
    'hybrid_n_correct_hard',
    'hybrid_n_reject',
    'hybrid_n_correct_total',
    'n_easy',
    'n_hard',
    'n_very_hard',
    'n_total',
]
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

experiment_name = '5x10_Sep10/20250910-221425'


def load_results():
    parsed_data = {}
    results_path = results_root / experiment_name
    for dataset in results_path.iterdir():
        if not dataset.is_dir():
            continue
        for results_file in dataset.iterdir():
            run_info = parse_filename(results_file.name)
            algorithm_type = run_info.algorithm_type_hash()
            if algorithm_type not in parsed_data:
                parsed_data[algorithm_type] = {}
            if dataset.name not in parsed_data[algorithm_type]:
                parsed_data[algorithm_type][dataset.name] = {
                    'train': [],
                    'test': [],
                }
            results_info = parse_results_file(results_file)
            results_info['run_id'] = run_info.run_id
            parsed_data[algorithm_type][dataset.name][run_info.train_or_test].append(results_info)
    return parsed_data


def bulk_text_report(reports_path):
    raw_data = {}
    results_path = results_root / experiment_name
    for dataset in results_path.iterdir():
        if not dataset.is_dir():
            continue
        for results_file in dataset.iterdir():
            run_info = parse_filename(results_file.name)
            algorithm_type = run_info.algorithm_type_hash()
            if algorithm_type not in raw_data:
                raw_data[algorithm_type] = {}
            if dataset.name not in raw_data[algorithm_type]:
                raw_data[algorithm_type][dataset.name] = {
                    'train': {key: 0 for key in results_keys},
                    'test': {key: 0 for key in results_keys},
                }
            results_info = parse_results_file(results_file)
            if run_info.train_or_test == 'train':
                add_dicts(raw_data[algorithm_type][dataset.name]['train'], results_info)
            else:
                add_dicts(raw_data[algorithm_type][dataset.name]['test'], results_info)

    for k, v in raw_data.items():
        report = ''
        for dataset_name, train_test in v.items():
            report += f'DATASET ID: {dataset_name}\n'
            report += '--TRAIN--\n'
            report += results_dict_to_text(v[dataset_name]['train'])
            report += '--TEST--\n'
            report += results_dict_to_text(v[dataset_name]['test'])
            report += '\n'
        report_path = f'{reports_path / k}.txt'
        with open(report_path, 'w') as f:
            f.write(report)


def bulk_latex_report(train_or_test):
    report = ''
    results = load_results()
    bg_key = 'decision_tree-xgboost-decision_tree-binary'
    tg_key = 'decision_tree-xgboost-decision_tree-ternary'
    assert(results[bg_key].keys() == results[tg_key].keys())
    dataset_ids = results[bg_key].keys()
    for dataset_id in dataset_ids:
        if dataset_id in dataset_names:
            dset_name = dataset_names[dataset_id]
        else:
            dset_name = dataset_id
        report += f'{dset_name} & '
        binary_grader = latex_df_from_dict(results[bg_key][dataset_id][train_or_test])
        ternary_grader = latex_df_from_dict(results[tg_key][dataset_id][train_or_test])

        report += f'{(100 * binary_grader["glass_accuracy"].mean()):.2f} & '
        report += f'± {(100 * binary_grader["glass_accuracy"].std()):.2f} & '

        report += f'{(100 * binary_grader["black_accuracy"].mean()):.2f} & '
        report += f'± {(100 * binary_grader["black_accuracy"].std()):.2f} & '

        report += f'{(100 * binary_grader["hybrid_accuracy"].mean()):.2f} & '
        report += f'± {(100 * binary_grader["hybrid_accuracy"].std()):.2f} & '

        report += f'{(100 * ternary_grader["hybrid_accuracy"].mean()):.2f} & '
        report += f'± {(100 * ternary_grader["hybrid_accuracy"].std()):.2f} & '

        report += f'{(100 * ternary_grader["glass_usage"].mean()):.2f} & '
        report += f'± {(100 * ternary_grader["glass_usage"].std()):.2f} & '

        report += f'{(100 * ternary_grader["black_usage"].mean()):.2f} & '
        report += f'± {(100 * ternary_grader["black_usage"].std()):.2f} & '

        report += f'{(100 * ternary_grader["reject_rate"].mean()):.2f} & '
        report += f'± {(100 * ternary_grader["reject_rate"].std()):.2f}\\\\ \\hline'
        report += '\n'
    print(dataset_ids)
    print(report)
    return


def latex_df_from_dict(results_dict):
    df = pd.DataFrame.from_dict(results_dict).set_index('run_id')
    df['glass_accuracy'] = df['glass_n_correct_total'] / df['n_total']
    df['black_accuracy'] = df['black_n_correct_total'] / df['n_total']
    df['hybrid_accuracy'] = df['hybrid_n_correct_total'] / (df['n_easy'] + df['n_hard'])
    df['glass_usage'] = df['n_easy'] / df['n_total']
    df['black_usage'] = df['n_hard'] / df['n_total']
    df['reject_rate'] = df['n_very_hard'] / df['n_total']
    return df

    # brst-w & 96.56 & ± 1.00 & 99.86 & ± 0.23 & 99.74 & ± 0.30 & 23.32 & ± 8.39 & 23.32 & ± 8.39 & 23.32 & ± 8.39 & 23.32 & ± 8.39\\ \hline
#     for algo_type, v in results.items():
#         for dataset_id, train_test in v.items():
#             train_results = train_test['train']
#             df = pd.DataFrame(train_results).set_index('run_id')
#             df['hybrid_accuracy'] = df['hybrid_n_correct_total'] / df['n_total']
#             print(df['hybrid_accuracy'].mean())
#             print(df['hybrid_accuracy'].std())
#             return
# #    print(results)


def main():
    reports_path = reports_root / experiment_name
    reports_path.mkdir(exist_ok=True, parents=True)
    bulk_latex_report('train')
    bulk_latex_report('test')
    print('OK')


@dataclass
class RunInfo:
    run_id: int
    glass_box_type: str
    black_box_type: str
    grader_type: str
    train_or_test: str
    grader_degree: str

    def algorithm_type_hash(self):
        return f'{self.glass_box_type}-{self.black_box_type}-{self.grader_type}-{self.grader_degree}'


def parse_filename(filename):
    bits = filename[:-4].split('-')
    return RunInfo(
        run_id=int(bits[0]),
        glass_box_type=bits[1],
        black_box_type=bits[2],
        grader_type=bits[3],
        train_or_test=bits[4],
        grader_degree=bits[5],
    )


def add_dicts(target_dict, addend_dict):
    for key in results_keys:
        assert key in target_dict
        assert key in addend_dict
        target_dict[key] += addend_dict[key]


if __name__ == '__main__':
    main()
