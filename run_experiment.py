import argparse
import datetime
import json

from pathlib import Path
from functools import partial

from src.experiment import ExperimentClassification
from src.lib import Settings, write_git_info
from src.models import OptunaXGBoost, OptunaXGBoostClassifier, tuned_decision_tree_classifier

dataset_baskets = {
    'fast':     [17, 19, 43, 151, 176, 212, 451, 545, 563, 863],
    'midscope': [
        17, 19, 43, 52, 94, 96, 151, 174, 176, 212, 267,
        329, 451, 519, 537, 545, 563, 850, 863, 887, 891
    ],
    'complete': [
        17,  # Breast Cancer Wisconsin (Diagnostic)
        19,  # Car Evaluation
        43,  # Haberman's Survival
        52,  # Ionosphere
        59,  # Letter Recognition
        78,  # Page Blocks Classification
        94,  # Spambase
        96,  # SPECTF Heart
        151,  # Connectionist Bench (Sonar, Mines vs. Rocks)
        159,  # MAGIC Gamma Telescope
        174,  # Parkinsons
        176,  # Blood Transfusion Service Center
        212,  # Vertebral Column
        267,  # Banknote authentication
        329,  # Diabetic Retinopathy Debrecen
        372,  # HTRU2
        451,  # Breast Cancer Coimbra
        519,  # Heart Failure Clinical Records
        537,  # Cervical Cancer Behavior Risk
        545,  # Rice (Cammeo and Osmancik)
        563,  # Iranian Churn
        572,  # Taiwanese Bankruptcy Prediction
        602,  # Dry Bean
        722,  # NATICUSdroid (Android Permissions)
        827,  # Sepsis Survival Minimal Clinical Records
        850,  # Raisin
        863,  # Maternal health risk
        887,  # National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset
        890,  # AIDS Clinical Trials Group Study 175
        891,  # CDC Diabetes Health Indicators
    ],
}

def main(s: Settings):
    print("Starting Experiment...")

    output_path = Path(f'./output/results/{s.experiment_slug}') / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path.mkdir(parents=True)
    with open(output_path / 'settings.json', 'w') as f:
        json.dump(settings.__dict__, f)
    with open(output_path / 'info.txt', 'w') as f:
        write_git_info(f)

    for dataset_id in s.datasets:
        print(f'.... {dataset_id} ...')

        glass_box_choices = {
            'decision_tree_nonleaked': partial(tuned_decision_tree_classifier, max_allowed_depth=4)
        }

        black_box_choices = {
            'xgboost_noleak_100trials': partial(OptunaXGBoostClassifier, n_trials=100, n_jobs=settings.n_jobs),
        }

        glass_box_grader = partial(tuned_decision_tree_classifier, max_allowed_depth=4)
        grey_box_grader = partial(tuned_decision_tree_classifier, max_allowed_depth=16)
        grader_choices = {
            'dt': glass_box_grader,
            'grey': grey_box_grader,
        }

        exp = ExperimentClassification(glass_box_choices, black_box_choices, grader_choices, settings)
        exp.run_experiment(dataset_id, output_path)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n-splits', type=int, default=10)
    parser.add_argument('-n-repeats', type=int, default=5)
    parser.add_argument('-cw-n-splits', type=int, default=5)
    parser.add_argument('-cw-n-repeats', type=int, default=1)
    parser.add_argument('-cw-stop-condition', type=str, default='dynamic', choices=['dynamic', 'static'])
    parser.add_argument('--experiment_slug', type=str, default='Experiment_')
    parser.add_argument('--dataset-basket', type=str, choices=dataset_baskets.keys())
    parser.add_argument('--dataset', type=str, action='append')
    args = parser.parse_args()

    if args.dataset_basket is not None:
        datasets = dataset_baskets[args.dataset_basket]
    elif args.dataset is not None:
        datasets = args.dataset
    else:
        datasets = []

    settings = Settings(
        experiment_slug=args.experiment_slug,
        datasets=datasets,
        n_repeats=args.n_repeats,
        n_splits=args.n_splits,
        cw_n_splits=args.cw_n_splits,
        cw_n_repeats=args.cw_n_repeats,
        cw_stop_condition=args.cw_stop_condition,
        n_jobs=-1,
    )

    print(settings)
    main(settings)
