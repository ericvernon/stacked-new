import argparse

import shutil
from pathlib import Path

from src.reporting import load_results_from_path, parse_results_df, bulk_text_report

reports_root = Path('./output/reports')
results_root = Path('./output/results')


def main(experiment_slug, experiment_id):
    results_path = results_root / experiment_slug / experiment_id
    reports_path = reports_root / experiment_slug / experiment_id
    reports_path.mkdir(exist_ok=True, parents=True)
    results = load_results_from_path(results_path, only_load=['17'])
    text_report = bulk_text_report(
        results['decision_tree-xgboost-dt-double'],
        'double-GB'
    )
    print(text_report)
    # x = parse_results_df(
    #         results['decision_tree-xgboost-dt-double']['17']['test'][4],
    #         'double-GB'
    # )
    # print(x)
#    bulk_text_report(reports_path, results)
#    csv_summary_report(reports_path / 'summary.txt', results, algo_prefix='GB_Prio_5xInf15_')
    # bulk_latex_report('train')
    # bulk_latex_report('test')
    shutil.copy(Path(results_path / 'info.txt'), Path(reports_path / 'info.txt'))
    shutil.copy(Path(results_path / 'settings.json'), Path(reports_path / 'settings.json'))
    print('OK')






default_slug = 'Oct21_Fast_Subset_Main3x10_CW5x1'
default_id = '20251021-152154'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-slug', type=str, default=default_slug)
    parser.add_argument('--exp-id', type=str, default=default_id)
    args = parser.parse_args()
    main(args.exp_slug, args.exp_id)
