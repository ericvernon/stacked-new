import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from collections import deque
from imblearn.over_sampling import SMOTE, RandomOverSampler
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from .lib import Settings, DIFFICULTY_EASY, DIFFICULTY_HARD, DIFFICULTY_VERY_HARD
from .data import load_dataset


def save_results(path, X, y_truth, glass_box_model, black_box_model, grader_model, grader_model2=None, save_X=False):
    n_features = X.shape[1]
    if save_X:
        df = pd.DataFrame(X, index=None, columns=[f'X_{i}' for i in range(n_features)])
    else:
        df = pd.DataFrame()
    df['y_truth'] = y_truth
    df['y_glass'] = glass_box_model.predict(X)
    df['y_black'] = black_box_model.predict(X)
    df['y_grader'] = grader_model.predict(X)
    if grader_model2 is not None:
        df['y_grader2'] = grader_model2.predict(X)
    df.to_csv(path, index=False)


class Experiment(ABC):
    def __init__(self, glass_box_choices: dict, black_box_choices: dict, grader_choices: dict, settings: Settings):
        self._split_fn = None
        self._glass_box_choices = glass_box_choices
        self._black_box_choices = black_box_choices
        self._grader_choices = grader_choices
        self._settings = settings

    def run_experiment(self, dataset_id, output_folder: Path):
        result_path = output_folder / str(dataset_id)
        result_path.mkdir()

        # Encode class names - we assume all labels are known beforehand (i.e. fit to whole dataset)
        X, y = load_dataset(dataset_id)
        le = LabelEncoder()
        y = le.fit_transform(y)

        splitter = self._split_fn(n_repeats=self._settings.n_repeats, n_splits=self._settings.n_splits, random_state=0)
        for data_split_idx, (train_index, test_index) in enumerate(
                tqdm(splitter.split(X, y), total=(self._settings.n_repeats*self._settings.n_splits)),
        ):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # The glass/black box models are independent of the grader --
            # Train + save (in memory) these first, so we can test them with multiple graders without retraining
            glass_box_models = {}
            for name, model_fn in self._glass_box_choices.items():
                model = model_fn()
                model_wrong_idx = collect_wrong_indices(model_fn, X_train, y_train, self._settings.cw_n_splits,
                                                        self._settings.cw_n_repeats, self._settings.cw_stop_condition)
                model.fit(X_train, y_train)
                glass_box_models[name] = {
                    'function': model_fn,
                    'wrong_idx': model_wrong_idx,
                    'trained_model': model,
                }

            black_box_models = {}
            for name, model_fn in self._black_box_choices.items():
                model = model_fn()
                model_wrong_idx = collect_wrong_indices(model_fn, X_train, y_train, self._settings.cw_n_splits,
                                                        self._settings.cw_n_repeats, self._settings.cw_stop_condition)
                model.fit(X_train, y_train)
                black_box_models[name] = {
                    'function': model_fn,
                    'wrong_idx': model_wrong_idx,
                    'trained_model': model,
                }

            # Now loop through all combinations we're interested in
            # One option would be to precompute the graders, which makes the nested for loops a little cleaner,
            #   but since the grades are dependent on the base classifiers, that
            #   introduces potentially annoying/confusing data structures for caching them...
            for glass_box_name, glass_box_model_info in glass_box_models.items():
                for grader_name, grader_fn in self._grader_choices.items():
                    # The binary grader only depends on the glass box, so train it once
                    b_grader_x, b_grader_y = get_binary_grader_data(glass_box_model_info['wrong_idx'], X_train)
                    binary_grader_model = grader_fn(n_jobs=self._settings.n_jobs)
                    binary_grader_model.fit(b_grader_x, b_grader_y)

                    for black_box_name, black_box_model_info in black_box_models.items():
                        # The ternary grader is dependent on *both* base classifiers
                        t_grader_x, t_grader_y = get_ternary_grader_data(glass_box_model_info['wrong_idx'],
                                                                              black_box_model_info['wrong_idx'],
                                                                              X_train)
                        ternary_grader_model = grader_fn(n_jobs=self._settings.n_jobs)
                        ternary_grader_model.fit(t_grader_x, t_grader_y)

                        accept_reject_x, accept_reject_y = get_binary_grader_data(black_box_model_info['wrong_idx'],
                                                                                  X_train)
                        accept_reject_grader = grader_fn(n_jobs=self._settings.n_jobs)
                        accept_reject_grader.fit(accept_reject_x, accept_reject_y)

                        save_results(
                            result_path /
                            f'{data_split_idx}-{glass_box_name}-{black_box_name}-{grader_name}-train-binary.txt',
                            X_train, y_train, glass_box_model_info['trained_model'],
                            black_box_model_info['trained_model'], binary_grader_model, save_X=self._settings.save_X
                        )

                        save_results(
                            result_path /
                            f'{data_split_idx}-{glass_box_name}-{black_box_name}-{grader_name}-test-binary.txt',
                            X_test, y_test, glass_box_model_info['trained_model'],
                            black_box_model_info['trained_model'], binary_grader_model, save_X=self._settings.save_X
                        )

                        save_results(
                            result_path /
                            f'{data_split_idx}-{glass_box_name}-{black_box_name}-{grader_name}-train-ternary.txt',
                            X_train, y_train, glass_box_model_info['trained_model'],
                            black_box_model_info['trained_model'], ternary_grader_model, save_X=self._settings.save_X
                        )

                        save_results(
                            result_path /
                            f'{data_split_idx}-{glass_box_name}-{black_box_name}-{grader_name}-test-ternary.txt',
                            X_test, y_test, glass_box_model_info['trained_model'],
                            black_box_model_info['trained_model'], ternary_grader_model, save_X=self._settings.save_X
                        )

                        save_results(
                            result_path /
                            f'{data_split_idx}-{glass_box_name}-{black_box_name}-{grader_name}-train-double.txt',
                            X_train, y_train, glass_box_model_info['trained_model'], black_box_model_info['trained_model'],
                            binary_grader_model, accept_reject_grader, save_X=self._settings.save_X
                        )

                        save_results(
                            result_path /
                            f'{data_split_idx}-{glass_box_name}-{black_box_name}-{grader_name}-test-double.txt',
                            X_test, y_test, glass_box_model_info['trained_model'], black_box_model_info['trained_model'],
                            binary_grader_model, accept_reject_grader, save_X=self._settings.save_X
                        )


class ExperimentClassification(Experiment):
    def __init__(self, glass_box_choices: dict, black_box_choices: dict, grader_choices: dict, settings: Settings):
        super().__init__(glass_box_choices, black_box_choices, grader_choices, settings)
        self._split_fn = RepeatedStratifiedKFold


def get_binary_grader_data(wrong_idx, X_train):
    n_patterns = X_train.shape[0]
    n_wrong = len(wrong_idx)
    difficulty = np.full(shape=(n_patterns,), fill_value=DIFFICULTY_EASY)
    difficulty[np.array(list(wrong_idx))] = DIFFICULTY_HARD
    if n_wrong == 0 or n_wrong == X_train.shape[0]:
        return X_train.copy(), difficulty
    elif n_wrong == 1:
        os = RandomOverSampler(random_state=0)
        return os.fit_resample(X_train, difficulty)
    else:
        k_neighbors = min(n_wrong - 1, 5)
        smote = SMOTE(k_neighbors=k_neighbors, random_state=0)
        return smote.fit_resample(X_train, difficulty)


def get_ternary_grader_data(glass_box_wrong, black_box_wrong, X_train, skip_oversampling=False):
    # Patterns which the black box cannot classify are always very hard, regardless of the glass box
    #   (i.e. we use the following confusion matrix:)
    #                           Black Box
    #                           Correct         Incorrect
    # Glass Box Correct         Easy            Very Hard
    #           Incorrect       Hard            Very Hard
    #
    n_patterns = X_train.shape[0]
    n_set = set(np.arange(n_patterns))

    difficulty = np.full(shape=(n_patterns,),  fill_value=DIFFICULTY_EASY, dtype=np.int64)

    black_box_correct = n_set.difference(black_box_wrong)
    hard_set = black_box_correct.intersection(glass_box_wrong)
    if len(hard_set) > 0:
        difficulty[np.array(list(hard_set))] = DIFFICULTY_HARD

    very_hard_set = glass_box_wrong.intersection(black_box_wrong)
    if len(very_hard_set) > 0:
        difficulty[np.array(list(very_hard_set))] = DIFFICULTY_VERY_HARD

    bins = np.bincount(difficulty)
    if bins.size < 2 or skip_oversampling:
        # This can happen if all patterns are "easy"... bincount will return like [100], instead of [100, 0, 0]
        return X_train.copy(), difficulty

    min_bin = np.min(bins)
    if min_bin < 2:  # i.e. there exists a bin with less than 2 patterns, which makes SMOTE impossible
        os = RandomOverSampler(random_state=0)
        return os.fit_resample(X_train, difficulty)
    else:
        k_neighbors = min(min_bin - 1, 5)
        smote = SMOTE(k_neighbors=k_neighbors, random_state=0)
        return smote.fit_resample(X_train, difficulty)


# Stop conditions:
    # "fixed"   - Stop after n_splits x n_repeats, no matter what
    # "dynamic" - Perform n_splits CV infinitely, until n_repeats complete
def collect_wrong_indices(model_function, X_train, y_train, n_splits, n_repeats, stop_condition='dynamic'):
    if stop_condition == 'dynamic':
        kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=555_555, random_state=0)
    elif stop_condition == 'static':
        kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    else:
        raise ValueError('Stop condition must be either "dynamic" or "static"')

    q_len = n_splits * n_repeats
    q = deque([1] * q_len, maxlen=q_len)
    all_incorrect_idx = set()
    for data_split_idx, (train_idx, calibration_idx) in enumerate(kfold.split(X_train, y_train)):
        model = model_function()
        model.fit(X_train[train_idx], y_train[train_idx])
        wrong_idx_within_calibration = model.predict(X_train[calibration_idx]) != y_train[calibration_idx]
        wrong_idx_within_training = calibration_idx[wrong_idx_within_calibration]
        n_before = len(all_incorrect_idx)
        all_incorrect_idx.update(wrong_idx_within_training)
        n_new = len(all_incorrect_idx) - n_before
        q.append(n_new)
        # Only evaluate stopping after we complete a full CV set
        # ... can consider changing this later + probably a minor detail, but for now do it this way
        if stop_condition == 'dynamic' and (data_split_idx + 1) % n_splits == 0 and sum(q) == 0:
            break
    return all_incorrect_idx
