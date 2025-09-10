# All the model definitions (including parameter tuning options) go here.
# This is not the best "clean code"... but it is sufficient for "academic code"!
import numpy as np
import optuna
from optuna.samplers import TPESampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

CV_N_SPLITS = 5
CV_N_REPEATS = 3


def tuned_decision_tree_classifier(max_allowed_depth, n_jobs=None):
    return GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=0),
        cv=RepeatedStratifiedKFold(n_repeats=CV_N_REPEATS, n_splits=CV_N_SPLITS, random_state=0),
        param_grid={
            "max_depth": np.arange(1, max_allowed_depth + 1),
        },
        n_jobs=n_jobs
    )


def random_forest_classifier(n_jobs=None):
    return GridSearchCV(
        estimator=RandomForestClassifier(random_state=0),
        cv=RepeatedStratifiedKFold(n_repeats=CV_N_REPEATS, n_splits=CV_N_SPLITS, random_state=0),
        param_grid={
            "max_depth": [4, 8, 16, 32, 64, 128],
            "n_estimators": [50, 100, 150, 200, 250],
        },
        n_jobs=n_jobs
    )


class OptunaXGBoost:
    def __init__(self, n_trials=500, n_jobs=1):
        self._model = None
        self._model_fn = None
        self._split_fn = None
        self._n_trials = n_trials
        self._n_jobs = n_jobs
        self._study = None

    def optimize(self, X, y):
        def fn(trial):
            max_depth = trial.suggest_int("max_depth", low=4, high=128)
            min_child_weight = trial.suggest_float("min_child_weight", low=0, high=4)
            gamma = trial.suggest_float("gamma", low=0, high=4)
            xgb = self._model_fn(
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                gamma=gamma,
                random_state=0,
            )
            cv = self._split_fn(n_repeats=CV_N_REPEATS, n_splits=CV_N_SPLITS, random_state=0)
            return cross_val_score(xgb, X, y, cv=cv).mean()

        sampler = TPESampler(seed=0)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(fn, n_trials=self._n_trials, n_jobs=self._n_jobs, show_progress_bar=True)
        self._study = study

    def fit(self, X, y):
        if self._study is None:
            self.optimize(X, y)
        params = self._study.best_params
        params['random_state'] = 0
        self._model = self._model_fn(**params)
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def score(self, X, y):
        return self._model.score(X, y)

    def get_study_results(self):
        assert (self._study is not None)
        return self._study.best_params, self._study.best_value


class OptunaXGBoostClassifier(OptunaXGBoost):
    def __init__(self, n_trials=500, n_jobs=1):
        super().__init__(n_trials, n_jobs)
        self._model_fn = XGBClassifier
        self._split_fn = RepeatedStratifiedKFold
