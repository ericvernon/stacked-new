import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from imblearn.over_sampling import SMOTE, RandomOverSampler
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from .lib import Settings, DIFFICULTY_EASY, DIFFICULTY_HARD, DIFFICULTY_VERY_HARD
from .data import load_dataset


def save_results_switching(path, X, y_truth, glass_box_model, black_box_model, switch_grader, reject_grader, save_X=False):
    n_features = X.shape[1]
    if save_X:
        df = pd.DataFrame(X, index=None, columns=[f'X_{i}' for i in range(n_features)])
    else:
        df = pd.DataFrame()
    df['y_truth'] = y_truth
    df['y_glass'] = glass_box_model.predict(X)
    df['y_black'] = black_box_model.predict(X)
    df['switch_grader'] = switch_grader.predict(X)
    df['reject_grader'] = reject_grader.predict(X)
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
        for data_split_idx, (train_index, test_index) in enumerate(splitter.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Scale X values. Currently using a simple MinMaxScaler,
            # which just moves values into the range [0, 1]
            mms = MinMaxScaler()
            mms.fit(X_train)
            X_train, X_test = mms.transform(X_train), mms.transform(X_test)

            X_subtrain, X_calibration, y_subtrain, y_calibration = train_test_split(
                X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)

            # The glass/black box models are independent of the grader --
            # Train + save (in memory) these first, so we can test them with multiple graders without retraining
            glass_box_models = {}
            for name, model_fn in self._glass_box_choices.items():
                model = model_fn(n_jobs=self._settings.n_jobs)
                model.fit(X_subtrain, y_subtrain)
                glass_box_models[name] = model

            black_box_models = {}
            for name, model_fn in self._black_box_choices.items():
                model = model_fn(n_jobs=self._settings.n_jobs)
                model.fit(X_subtrain, y_subtrain)
                black_box_models[name] = model

            # Now loop through all combinations we're interested in
            # One option would be to precompute the graders, which makes the nested for loops a little cleaner,
            #   but since the grades are dependent on the base classifiers, that
            #   introduces potentially annoying/confusing data structures for caching them...
            for glass_box_name, glass_box_model in glass_box_models.items():
                for grader_name, grader_fn in self._grader_choices.items():
                    # The binary grader only depends on the glass box, so train it once
                    switch_grader_x, switch_grader_y = self.get_grader_data(glass_box_model, X_calibration, y_calibration)
                    switch_grader = grader_fn(n_jobs=self._settings.n_jobs)
                    switch_grader.fit(switch_grader_x, switch_grader_y)

                    for black_box_name, black_box_model in black_box_models.items():

                        reject_grader_x, reject_grader_y = self.get_grader_data(black_box_model, X_calibration, y_calibration)
                        reject_grader = grader_fn(n_jobs=self._settings.n_jobs)
                        reject_grader.fit(reject_grader_x, reject_grader_y)

                        save_results_switching(
                            result_path /
                            f'{data_split_idx}-{glass_box_name}-{black_box_name}-{grader_name}-train-ternary.txt',
                            X_train, y_train, glass_box_model, black_box_model, switch_grader, reject_grader,
                            self._settings.save_X
                        )

                        save_results_switching(
                            result_path /
                            f'{data_split_idx}-{glass_box_name}-{black_box_name}-{grader_name}-test-ternary.txt',
                            X_test, y_test, glass_box_model, black_box_model, switch_grader, reject_grader,
                            self._settings.save_X
                        )

    @abstractmethod
    def get_grader_data(self, glass_box_model, X_train, y_train):
        """
        Get data for the basic, 2-class grader
        0: Easy (Use b_easy)
        1: Hard (Use b_hard)
        """
        pass

    @abstractmethod
    def get_reject_grader_data(self, glass_box_model, black_box_model, X_train, y_train):
        """
        Get data for the 3-class grader extension
        0: Easy (Use b_easy)
        1: Hard (Use b_hard)
        2: Very hard (Reject)
        """
        pass


class ExperimentClassification(Experiment):
    def __init__(self, glass_box_choices: dict, black_box_choices: dict, grader_choices: dict, settings: Settings):
        super().__init__(glass_box_choices, black_box_choices, grader_choices, settings)
        self._split_fn = RepeatedStratifiedKFold

    def get_grader_data(self, glass_box_model, X_train, y_train):
        predict_train = glass_box_model.predict(X_train)
        wrong_idx = (predict_train != y_train).astype(int)

        n_wrong = np.count_nonzero(wrong_idx)
        if n_wrong == 0 or n_wrong == X_train.shape[0]:
            return X_train.copy(), wrong_idx
        elif n_wrong == 1:
            os = RandomOverSampler(random_state=0)
            return os.fit_resample(X_train, wrong_idx)
        else:
            k_neighbors = min(n_wrong - 1, 5)
            smote = SMOTE(k_neighbors=k_neighbors, random_state=0)
            return smote.fit_resample(X_train, wrong_idx)

    def get_reject_grader_data(self, glass_box_model, black_box_model, X_train, y_train):
        # Patterns which the black box cannot classify are always very hard, regardless of the glass box
        #   (i.e. we use the following confusion matrix:)
        #                           Black Box
        #                           Correct         Incorrect
        # Glass Box Correct         Easy            Very Hard
        #           Incorrect       Hard            Very Hard
        #
        n = X_train.shape[0]
        results = np.full((n,),  fill_value=DIFFICULTY_EASY, dtype=np.int64)

        predict_easy = glass_box_model.predict(X_train)
        hard_idx = predict_easy != y_train
        results[hard_idx] = DIFFICULTY_HARD

        predict_hard = black_box_model.predict(X_train)
        very_hard_idx = predict_hard != y_train
        results[very_hard_idx] = DIFFICULTY_VERY_HARD

        bins = np.bincount(results)
        if bins.size < 2:
            # This can happen if all patterns are "easy"... bincount will return like [100], instead of [100, 0, 0]
            return X_train.copy(), results

        min_bin = np.min(bins)
        if min_bin < 2:  # i.e. there exists a bin with less than 2 patterns, which makes SMOTE impossible
            os = RandomOverSampler(random_state=0)
            return os.fit_resample(X_train, results)
        else:
            k_neighbors = min(min_bin - 1, 5)
            smote = SMOTE(k_neighbors=k_neighbors)
            return smote.fit_resample(X_train, results)


class ExperimentRegression(Experiment):
    def __init__(self, glass_box_choices: dict, black_box_choices: dict, grader_choices: dict, settings: Settings):
        super().__init__(glass_box_choices, black_box_choices, grader_choices, settings)
        self._split_fn = RepeatedKFold

    def get_grader_data(self, glass_box_model, X_train, y_train, percentile=50):
        predict_train = glass_box_model.predict(X_train)
        square_error = np.square(predict_train - y_train)
        hard_threshold = np.percentile(square_error, percentile)
        is_hard = (square_error > hard_threshold).astype(int)

        # This will just add 0-1 synthetic "hard" patterns...
        smote = SMOTE(k_neighbors=5, random_state=0)
        x2, y2 = smote.fit_resample(X_train, is_hard)
        return x2, y2

    def get_reject_grader_data(self, glass_box_model, black_box_model, X_train, y_train, percentile=50):
        n = X_train.shape[0]
        results = np.zeros((n,), dtype=np.int64)

        glass_box_predictions = glass_box_model.predict(X_train)
        gb_square_error = np.square(glass_box_predictions - y_train)
        gb_threshold = np.percentile(gb_square_error, percentile)

        black_box_predictions = black_box_model.predict(X_train)
        bb_square_error = np.square(black_box_predictions - y_train)
        bb_threshold = np.percentile(bb_square_error, percentile)

        hard_idx = gb_square_error > gb_threshold
        results[hard_idx] = DIFFICULTY_HARD

        vhard_idx = bb_square_error > bb_threshold
        results[vhard_idx] = DIFFICULTY_VERY_HARD

        bins = np.bincount(results)
        if bins.size < 2:
            # This can happen if all patterns are "easy"... bincount will return like [100], instead of [100, 0, 0]
            return X_train.copy(), results

        min_bin = np.min(bins)
        if min_bin < 2:  # i.e. there exists a bin with less than 2 patterns, which makes SMOTE impossible
            os = RandomOverSampler(random_state=0)
            return os.fit_resample(X_train, results)
        else:
            k_neighbors = min(min_bin - 1, 5)
            smote = SMOTE(k_neighbors=k_neighbors)
            return smote.fit_resample(X_train, results)
