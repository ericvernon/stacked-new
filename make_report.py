import argparse

import shutil
from pathlib import Path

from src.reporting import (load_results_from_path, parse_results_dict, bulk_text_report, csv_summary_report,
                           difficulty_band_summary_latex, results_overview_latex, CSV_SUMMARY_HEADER)

reports_root = Path('./output/reports')
results_root = Path('./output/results')


def main(experiment_slug, experiment_id, calibration_name):
    results_path = results_root / experiment_slug / experiment_id
    reports_path = reports_root / experiment_slug / experiment_id
    reports_path.mkdir(exist_ok=True, parents=True)
    results = load_results_from_path(results_path)

    shutil.copy(Path(results_path / 'info.txt'), Path(reports_path / 'info.txt'))
    shutil.copy(Path(results_path / 'settings.json'), Path(reports_path / 'settings.json'))

    #  The main table of results
    results_proposed_method = parse_results_dict(results['decision_tree-xgboost-dt-double'], 'double-GB')
    results_binary_grader = parse_results_dict(results['decision_tree-xgboost-dt-binary'], 'binary')
    with (open(reports_path / 'main_table_latex_train.txt', 'w', encoding='UTF-8') as train_fh,
          open(reports_path / 'main_table_latex_test.txt', 'w', encoding='UTF-8') as test_fh):
        main_results_train, main_results_test = results_overview_latex(results_proposed_method, results_binary_grader)
        train_fh.write(main_results_train)
        test_fh.write(main_results_test)

    #  Table with detailed results for each difficulty band.
    with (open(reports_path / 'band_summary_train.txt', 'w', encoding='UTF-8') as train_fh,
          open(reports_path / 'band_summary_test.txt', 'w', encoding='UTF-8') as test_fh):
        band_summary_train, band_summary_test = difficulty_band_summary_latex(results_proposed_method)
        train_fh.write(band_summary_train)
        test_fh.write(band_summary_test)

    #  Summary statistics to compare grader variants
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
    print('OK')


experiments = [
    ['CW_5x5_Static_Midscope', '20251218-162132', '5x5_Static'],
    ['CW_5x5_Dynamic_Midscope','20251216-190523', '5x5_Dynamic'],
    ['CW_3x5_Static_Midscope', '20251218-162436', '3x5_Static'],
    ['CW_3x5_Dynamic_Midscope', '20251216-190628', '3x5_Dynamic'],
    ['CW_1x5_Static_Midscope', '20251218-174146', '1x5_Static'],
    ['CW_1x5_Dynamic_Complete', '20251212-182251', '1x5_Dynamic'],
]


if __name__ == '__main__':
    for experiment in experiments:
        main(experiment[0], experiment[1], experiment[2])
