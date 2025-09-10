import json

from datetime import datetime
from pathlib import Path
from src.data import load_dataset
from src.models import OptunaXGBoostClassifier, tuned_decision_tree_classifier
from sklearn.preprocessing import LabelEncoder

current_dir = Path(__file__).resolve().parent
cache_path = current_dir / '../data/param_cache'
n_trials = 500
decision_tree_max_allowed_depth = 4


def parameter_lookup(dataset_id):
    assert (cache_path.exists())
    if not (cache_path / f'{dataset_id}.json').exists():
        print(f"No cached parameters found for dataset {dataset_id}, optimizing...")
        optimize_and_save(dataset_id)
    return load_from_cache(dataset_id)


def optimize_and_save(dataset_id):
    X, y = load_dataset(dataset_id)
    le = LabelEncoder()
    y = le.fit_transform(y)
    xgb = OptunaXGBoostClassifier(n_trials=n_trials)
    xgb.optimize(X, y)
    xgb_params, xgb_accuracy = xgb.get_study_results()

    dt = tuned_decision_tree_classifier(max_allowed_depth=decision_tree_max_allowed_depth)
    dt.fit(X, y)
    # The stock JSON encoder can't handle np types
    dt_best_max_depth, dt_accuracy = int(dt.best_params_['max_depth']), float(dt.best_score_)

    result = {
        'xgboost': {
            'params': xgb_params,
            'accuracy': xgb_accuracy,
            'n_trials': n_trials,
        },
        'decision_tree': {
            'params': {
                'max_depth': dt_best_max_depth,
            },
            'accuracy': dt_accuracy,
            'max_allowed_depth': decision_tree_max_allowed_depth,
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(cache_path / f'{dataset_id}.json', 'w') as f:
        json.dump(result, f)


def load_from_cache(dataset_id):
    assert (cache_path / f'{dataset_id}.json').exists()
    with open(cache_path / f'{dataset_id}.json', 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    print(
        parameter_lookup(17),   # Breast Cancer Diagnosis (Wisconsin)
        parameter_lookup(19),   # Car Evaluation
        parameter_lookup(43),   # Haberman's Survival
        parameter_lookup(52),   # Ionosphere
        parameter_lookup(94),   # Spambase
        parameter_lookup(267),  # Banknote Authentication
        parameter_lookup(863),  # Maternal Health Risk
        # dataset_ids = [17, 43, 52, 94, 96, 151, 174, 176, 267]
    )
