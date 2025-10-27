import pandas as pd
import shutil

from dataclasses import dataclass
from pathlib import Path

from src.reporting import parse_results_file, bulk_text_report, csv_summary_report

reports_root = Path('./output/reports/GlassBlack_Prio')
results_root = Path('./output/results')


#experiment_name = 'Oct23_Fast_Subset_Main3x10_CW_25x5/20251024-002540'
#experiment_name = 'Oct23_Fast_Subset_Main3x10_CW_Infx5/20251024-020343'
experiment_name = 'Oct23_Fast_Subset_Main3x10_CW_Infx15/20251024-031112'


def load_results(only_load=None):
    parsed_data = {}
    results_path = results_root / experiment_name
    for dataset in results_path.iterdir():
        if not dataset.is_dir():
            continue
        if only_load is not None and dataset.name not in only_load:
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
            results_info = parse_results_file(results_file, grader_type=run_info.grader_type)
            results_info['run_id'] = run_info.run_id
            parsed_data[algorithm_type][dataset.name][run_info.train_or_test].append(results_info)
    return parsed_data


def parse_filename(filename):
    bits = filename[:-4].split('-')
    return RunInfo(
        run_id=int(bits[0]),
        glass_box_algo=bits[1],
        black_box_algo=bits[2],
        grader_algo=bits[3],
        train_or_test=bits[4],
        grader_type=bits[5],
    )

def main():
    reports_path = reports_root / experiment_name
    reports_path.mkdir(exist_ok=True, parents=True)
    results = load_results()
    bulk_text_report(reports_path, results)
    csv_summary_report(reports_path / 'summary.txt', results, algo_prefix='GB_Prio_5xInf15_')
    # bulk_latex_report('train')
    # bulk_latex_report('test')
    shutil.copy(Path(results_root / experiment_name / 'info.txt'), Path(reports_root / experiment_name / 'info.txt'))
    shutil.copy(Path(results_root / experiment_name / 'settings.json'), Path(reports_root / experiment_name / 'settings.json'))
    print('OK')


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



if __name__ == '__main__':
    main()
