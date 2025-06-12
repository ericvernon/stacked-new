import argparse
import datetime
import logging
import sys
import time

from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from classification import *
from models import *
from lib import log_startup, Settings

from src.experiment import ExperimentClassification, ExperimentRegression

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp-slug', type=str, default='exp_')
    parser.add_argument('-d', type=str, action='append')
    parser.add_argument('-n-repeats', type=int, default=3)
    parser.add_argument('-n-splits', type=int, default=10)
    parser.add_argument('-n-jobs', type=int, default=1)
    parser.add_argument('--regression', default=False, action='store_true')

    args = parser.parse_args()

    experiment_slug = args.exp_slug
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")

    output_folder = Path('../output/') / (experiment_slug + timestamp)
    output_folder.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        filename=output_folder / 'log.txt',
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    settings = Settings(
        n_repeats=args.n_repeats,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
    )

    datasets = args.d
    log_startup(logger, datasets, settings)

    glass_box_choices = {
        'ShallowDecisionTree': shallow_decision_tree_classifier,
    }

    black_box_choices = {
        'XGBoost': OptunaXGBoostClassifier,
        'RandomForest': random_forest_classifier,
    }

    grader_choices = {
        'ShallowDecisionTree': shallow_decision_tree_classifier,
        'XGBoost': OptunaXGBoostClassifier,
    }

    if args.regression:
        experiment = ExperimentRegression(glass_box_choices, black_box_choices, grader_choices, settings)
    else:
        experiment = ExperimentClassification(glass_box_choices, black_box_choices, grader_choices, settings)

    for dataset_id in datasets:
        logger.info("Starting Dataset: " + str(dataset_id))
        exp = ExperimentClassification(glass_box_choices, black_box_choices, grader_choices, settings)
        exp.run_experiment(dataset_id, output_folder)

    logger.info("Done")


if __name__ == '__main__':
    main()
