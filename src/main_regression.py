import datetime
import sys

from pathlib import Path

from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler

from data import load_dataset
from regression import *
from models import *
from lib import log_startup, Settings

logger = logging.getLogger(__name__)


def experiment_regression(dataset_ids: list, output_folder: Path, settings: Settings):
    log_startup(logger, dataset_ids, settings)
    for dataset_id in dataset_ids:
        logger.info("Starting Dataset: " + str(dataset_id))
        result_path = output_folder / str(dataset_id)
        result_path.mkdir()

        with (
            open(result_path / '_results_train_binary.txt', 'w', encoding='utf-8') as results_train_binary_fh,
            open(result_path / '_results_test_binary.txt', 'w', encoding='utf-8') as results_test_binary_fh,
            open(result_path / '_results_train_ternary.txt', 'w', encoding='utf-8') as results_train_ternary_fh,
            open(result_path / '_results_test_ternary.txt', 'w', encoding='utf-8') as results_test_ternary_fh,
        ):
            write_header_binary_grader(results_train_binary_fh)
            write_header_binary_grader(results_test_binary_fh)

            X, y = load_dataset(dataset_id)

            # Scale y to 0-1 ... it's okay to use the whole dataset here
            mms_y = MinMaxScaler()
            y = mms_y.fit_transform(y.reshape(-1, 1)).reshape(-1)

            rskf = RepeatedKFold(n_repeats=settings.n_repeats, n_splits=settings.n_splits, random_state=0)

            for data_split_idx, (train_index, test_index) in enumerate(rskf.split(X, y)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Scale X values. Currently using a simple MinMaxScaler,
                # which just moves values into the range [0, 1]
                mms = MinMaxScaler()
                mms.fit(X_train)
                X_train, X_test = mms.transform(X_train), mms.transform(X_test)

                white_box_fn = {
                    'ShallowDecisionTree': shallow_decision_tree_regressor,
                    'MediumDecisionTree': medium_decision_tree_regressor,
                }

                black_box_fn = {
                    'XGBoost': XGBRegressor,
                    'RandomForest': random_forest_regressor,
                }

                grader_fn = {
                    'ShallowDecisionTree': shallow_decision_tree_classifier,
                    'MediumDecisionTree': medium_decision_tree_classifier,
                    'XGBoost': OptunaXGBoost,
                    'RandomForest': random_forest_classifier,
                }

                # First train the black/white-box classifiers (including parameter tuning),
                # to avoid training multiple times within the for-loops.
                white_box_models = {}
                for name, fn in white_box_fn.items():
                    model = fn(n_jobs=settings.n_jobs)
                    model.fit(X_train, y_train)
                    white_box_models[name] = model

                black_box_models = {}
                for name, fn in black_box_fn.items():
                    model = fn(n_jobs=settings.n_jobs)
                    model.fit(X_train, y_train)
                    black_box_models[name] = model

                # Loop through all choices of white box, black box, grader
                for white_box_name, white_box_model in white_box_models.items():
                    for grader_name, fn in grader_fn.items():
                        b_grader_x, b_grader_y = get_binary_grader_data(white_box_model, X_train, y_train)
                        binary_grader_model = fn(n_jobs=settings.n_jobs)
                        binary_grader_model.fit(b_grader_x, b_grader_y)
                        for black_box_name, black_box_model in black_box_models.items():
                            results_train_binary = calculate_results_binary_grader(X_train, y_train, white_box_model,
                                                                                   black_box_model, binary_grader_model)
                            results_test_binary = calculate_results_binary_grader(X_test, y_test, white_box_model,
                                                                                  black_box_model, binary_grader_model)

                            save_results_binary_grader(results_train_binary_fh, data_split_idx, white_box_name,
                                                       black_box_name, grader_name, results_train_binary)

                            save_results_binary_grader(results_test_binary_fh, data_split_idx, white_box_name,
                                                       black_box_name, grader_name, results_test_binary)

                            t_grader_x, t_grader_y = get_ternary_grader_data(white_box_model, black_box_model, X_train, y_train)
                            ternary_grader_model = fn(n_jobs=settings.n_jobs)
                            ternary_grader_model.fit(t_grader_x, t_grader_y)

                            results_train_ternary = calculate_results_ternary_grader(X_train, y_train, white_box_model,
                                                                                     black_box_model, ternary_grader_model)
                            results_test_ternary = calculate_results_ternary_grader(X_train, y_train, white_box_model,
                                                                                    black_box_model, ternary_grader_model)

                            save_results_ternary_grader(results_train_ternary_fh, data_split_idx, white_box_name,
                                                        black_box_name, grader_name, results_train_ternary)
                            save_results_ternary_grader(results_test_ternary_fh, data_split_idx, white_box_name,
                                                        black_box_name, grader_name, results_test_ternary)

                            if settings.save_full_results:
                                save_full_results(
                                    result_path /
                                    f'{data_split_idx}-{white_box_name}-{black_box_name}-{grader_name}-train-binary.txt',
                                    X_train, y_train, white_box_model, black_box_model, binary_grader_model)

                                save_full_results(
                                    result_path /
                                    f'{data_split_idx}-{white_box_name}-{black_box_name}-{grader_name}-test-binary.txt',
                                    X_test, y_test, white_box_model, black_box_model, binary_grader_model)

                                save_full_results(
                                    result_path /
                                    f'{data_split_idx}-{white_box_name}-{black_box_name}-{grader_name}-train-ternary.txt',
                                    X_train, y_train, white_box_model, black_box_model, ternary_grader_model
                                )

                                save_full_results(
                                    result_path /
                                    f'{data_split_idx}-{white_box_name}-{black_box_name}-{grader_name}-test-ternary.txt',
                                    X_test, y_test, white_box_model, black_box_model, ternary_grader_model
                                )

    logger.info('Done')


def main():
    print(sys.argv)
    experiment_slug = 'regression_'
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
        n_repeats=5,
        n_splits=10,
        n_jobs=6,
        save_full_results=False,
    )

    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = []
    experiment_regression(datasets, output_folder, settings)


if __name__ == '__main__':
    main()
