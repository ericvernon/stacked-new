from dataclasses import dataclass
from pathlib import Path

from src.reporting import parse_results_file, text_report

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


def main():
    experiment_name = 'simple_experiment/20250625-151929'

    reports_path = reports_root / experiment_name
    reports_path.mkdir(exist_ok=True, parents=True)

    raw_data = {}
    results_path = results_root / experiment_name
    for dataset in results_path.iterdir():
        assert dataset.is_dir()
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
            report += text_report(v[dataset_name]['train'])
            report += '--TEST--\n'
            report += text_report(v[dataset_name]['test'])
            report += '\n'
        report_path = f'{reports_path / k}.txt'
        with open(report_path, 'w') as f:
            f.write(report)

    print('OK')


def add_dicts(target_dict, addend_dict):
    for key in results_keys:
        assert key in target_dict
        assert key in addend_dict
        target_dict[key] += addend_dict[key]


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


if __name__ == '__main__':
    main()
