import argparse

import shutil
from pathlib import Path

from src.reporting import (load_results_from_path, parse_results_dict, bulk_text_report, csv_summary_report,
                           CSV_SUMMARY_HEADER)

reports_root = Path('./output/reports')
results_root = Path('./output/results')


def main(experiment_slug, experiment_id):
    results_path = results_root / experiment_slug / experiment_id
    reports_path = reports_root / experiment_slug / experiment_id
    reports_path.mkdir(exist_ok=True, parents=True)
    results = load_results_from_path(results_path, only_load=['19'])

    infos = [
        ('decision_tree-xgboost-dt-binary', 'binary', 'Binary_Shallow'),
        ('decision_tree-xgboost-dt-double', 'double-AR', 'Double_AR_Shallow'),
        ('decision_tree-xgboost-dt-double', 'double-GB', 'Double_GB_Shallow'),
        ('decision_tree-xgboost-dt-ternary', 'ternary', 'Ternary_Shallow'),
        ('decision_tree-xgboost-grey-binary', 'binary', 'Binary_Deep'),
        ('decision_tree-xgboost-grey-double', 'double-AR', 'Double_AR_Deep'),
        ('decision_tree-xgboost-grey-double', 'double-GB', 'Double_GB_Deep'),
        ('decision_tree-xgboost-grey-ternary', 'ternary', 'Ternary_Deep')
    ]

    calibration_name = '1x5_Dynamic'
    with open(reports_path / 'summary.txt', 'w', encoding='UTF-8') as summary_fh:
        summary_fh.write(CSV_SUMMARY_HEADER + '\n')
        for result_key, grader_type, summary_name in infos:
            parsed_results = parse_results_dict(results[result_key], grader_type)
            summary = csv_summary_report(parsed_results, f'{calibration_name}_{summary_name}')
            summary_fh.write(summary)

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






default_slug = 'UltraFast_2x10_1x5_Dynamic'
default_id = '20251216-182246'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-slug', type=str, default=default_slug)
    parser.add_argument('--exp-id', type=str, default=default_id)
    args = parser.parse_args()
    main(args.exp_slug, args.exp_id)
