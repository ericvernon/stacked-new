# All the model definitions (including parameter tuning options) go here.
# This is not the best "clean code"... but it is sufficient for "academic code"!
import numpy as np
import optuna

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from xgboost import XGBClassifier, XGBRegressor


def shallow_decision_tree_classifier(n_jobs=None):
    return GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=19),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        param_grid={
            "max_depth": [1, 2, 3],
        },
        n_jobs=n_jobs
    )


def medium_decision_tree_classifier(n_jobs=None):
    return GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=19),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        param_grid={
            "max_depth": [1, 2, 3, 4, 5, 6],
        },
        n_jobs=n_jobs
    )


def shallow_decision_tree_regressor(n_jobs=None):
    return GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=19),
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        param_grid={
            "max_depth": [1, 2, 3],
        },
        n_jobs=n_jobs
    )


def medium_decision_tree_regressor(n_jobs=None):
    return GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=19),
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        param_grid={
            "max_depth": [1, 2, 3, 4, 5, 6],
        },
        n_jobs=n_jobs
    )


def random_forest_classifier(n_jobs=None):
    return GridSearchCV(
        estimator=RandomForestClassifier(random_state=19),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        param_grid={
            "max_depth": [4, 8, 16, 32, 64, 128],
            "n_estimators": [50, 100, 150, 200, 250],
        },
        n_jobs=n_jobs
    )


def random_forest_regressor(n_jobs=None):
    return GridSearchCV(
        estimator=RandomForestRegressor(random_state=19),
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        param_grid={
            "max_depth": [4, 8, 16, 32, 64, 128],
            "n_estimators": [50, 100, 150, 200, 250],
        },
        n_jobs=n_jobs
    )


class OptunaXGBoost:
    def __init__(self, n_trials=100, n_jobs=1, regression=False):
        self._model = None
        self._n_trials = n_trials
        self._n_jobs = n_jobs

        if regression:
            self._model_fn = XGBRegressor
            self._split_fn = KFold
        else:
            self._model_fn = XGBClassifier
            self._split_fn = StratifiedKFold

    def fit(self, X, y):
        def fn(trial):
            max_depth = trial.suggest_int("max_depth", low=4, high=128)
            min_child_weight = trial.suggest_float("min_child_weight", low=0, high=4)
            gamma = trial.suggest_float("gamma", low=0, high=4)
            xgb = self._model_fn(
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                gamma=gamma,
                random_state=19,
            )
            cv = self._split_fn(n_splits=5, shuffle=True, random_state=0)
            return cross_val_score(xgb, X, y, cv=cv).mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(fn, n_trials=self._n_trials, n_jobs=self._n_jobs)
        self._model = XGBClassifier(**study.best_params)
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)


def xgboost_classifier(n_jobs=None):
    return OptunaXGBoost(n_trials=100, n_jobs=n_jobs, regression=False)

def xgboost_regressor(n_jobs=None):
    return OptunaXGBoost(n_trials=100, n_jobs=n_jobs, regression=True)

# def xgboost_classifier(n_jobs=None):
#     return GridSearchCV(
#         estimator=XGBClassifier(),
#         cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
#         param_grid={
#             "max_depth": [4, 8, 16, 32, 64, 128],
#             "min_child_weight": np.logspace(start=0, stop=4, num=10),
#             "gamma": np.logspace(start=0, stop=4, num=10)
#         },
#         n_jobs=n_jobs
#     )
#
# def optimized_xgboost(X_train, y_train, classifier=True, n_trials=25, n_jobs=None):
#     if classifier:
#         cls = XGBClassifier
#     else:
#         cls = XGBRegressor
#
#     def fn(trial):
#         max_depth = trial.suggest_int("max_depth", low=1, high=128)
#         min_child_weight = trial.suggest_float("min_child_weight", low=0, high=8)
#         gamma = trial.suggest_float("gamma", low=0, high=4)
#         xgb = cls(
#             max_depth=max_depth,
#             min_child_weight=min_child_weight,
#             gamma=gamma,
#         )
#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
#         return cross_val_score(xgb, X_train, y_train, cv=cv).mean()
#
#     study = optuna.create_study(direction="maximize")
#     study.optimize(fn, n_trials=n_trials)
#     return cls(**study.best_params)