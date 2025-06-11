import argparse
import datetime
import logging
import sys
import time

from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from data import load_dataset
from classification import *
from models import *
from lib import log_startup, Settings

logger = logging.getLogger(__name__)


def experiment_classification(dataset_ids: list, output_folder: Path, settings: Settings):
    log_startup(logger, dataset_ids, settings)
    for dataset_id in dataset_ids:
        logger.info("Starting Dataset: " + str(dataset_id))
        result_path = output_folder / str(dataset_id)
        result_path.mkdir()

        X, y = load_dataset(dataset_id)

        # Encode class names - we assume all labels are known beforehand
        le = LabelEncoder()
        y = le.fit_transform(y)

        rskf = RepeatedStratifiedKFold(n_repeats=settings.n_repeats, n_splits=settings.n_splits, random_state=0)

        for data_split_idx, (train_index, test_index) in enumerate(rskf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Scale X values. Currently using a simple MinMaxScaler,
            # which just moves values into the range [0, 1]
            mms = MinMaxScaler()
            mms.fit(X_train)
            X_train, X_test = mms.transform(X_train), mms.transform(X_test)

            glass_box_choices = {
                'ShallowDecisionTree': shallow_decision_tree_classifier,
                # 'MediumDecisionTree': medium_decision_tree_classifier,
                # 'EXFuzzy': exfuzzy_classifier,
            }

            black_box_choices = {
                'XGBoost': OptunaXGBoostClassifier,
                'RandomForest': random_forest_classifier,
            }

            grader_choices = {
                'ShallowDecisionTree': shallow_decision_tree_classifier,
                # 'MediumDecisionTree': medium_decision_tree_classifier,
                # 'EXFuzzy': exfuzzy_classifier,
                'XGBoost': OptunaXGBoostClassifier,
                # 'RandomForest': random_forest_classifier,
            }

            # The glass/black box models are independent of the grader --
            # Train + save (in memory) these first so we can test them with multiple graders without retraining
            glass_box_models = {}
            for name, grader_fn in glass_box_choices.items():
                model = grader_fn(n_jobs=settings.n_jobs)
                model.fit(X_train, y_train)
                glass_box_models[name] = model

            black_box_models = {}
            for name, grader_fn in black_box_choices.items():
                model = grader_fn(n_jobs=settings.n_jobs)
                model.fit(X_train, y_train)
                black_box_models[name] = model

            # Now loop through all combinations we're interested in
            # The grader *is* dependent on both the glass and black box models, so we have to train a new one each time
            for glass_box_name, glass_box_model in glass_box_models.items():
                for grader_name, grader_fn in grader_choices.items():

                    # Binary grader (easy/hard) -- this is only dependent on the glass box, so only train once...
                    b_grader_x, b_grader_y = get_binary_grader_data(glass_box_model, X_train, y_train)
                    binary_grader_model = grader_fn(n_jobs=settings.n_jobs)
                    binary_grader_model.fit(b_grader_x, b_grader_y)

                    for black_box_name, black_box_model in black_box_models.items():
                        # Grader for easy/hard/very-hard (to support the reject option)
                        t_grader_x, t_grader_y = get_ternary_grader_data(glass_box_model, black_box_model, X_train, y_train)
                        ternary_grader_model = grader_fn(n_jobs=settings.n_jobs)
                        ternary_grader_model.fit(t_grader_x, t_grader_y)

                        save_results(
                            result_path /
                            f'{data_split_idx}-{glass_box_name}-{black_box_name}-{grader_name}-train-binary.txt',
                            X_train, y_train, glass_box_model, black_box_model, binary_grader_model
                        )

                        save_results(
                            result_path /
                            f'{data_split_idx}-{glass_box_name}-{black_box_name}-{grader_name}-test-binary.txt',
                            X_test, y_test, glass_box_model, black_box_model, binary_grader_model
                        )

                        save_results(
                            result_path /
                            f'{data_split_idx}-{glass_box_name}-{black_box_name}-{grader_name}-train-ternary.txt',
                            X_train, y_train, glass_box_model, black_box_model, ternary_grader_model
                        )

                        save_results(
                            result_path /
                            f'{data_split_idx}-{glass_box_name}-{black_box_name}-{grader_name}-test-ternary.txt',
                            X_test, y_test, glass_box_model, black_box_model, ternary_grader_model
                        )

    logger.info('Done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp-slug', type=str, default='exp_')
    parser.add_argument('-d', type=str, action='append')
    parser.add_argument('-n-repeats', type=int, default=3)
    parser.add_argument('-n-splits', type=int, default=10)
    parser.add_argument('-n-jobs', type=int, default=1)

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
    experiment_classification(datasets, output_folder, settings)


if __name__ == '__main__':
    main()
