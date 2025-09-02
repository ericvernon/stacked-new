import json

from datetime import datetime
from pathlib import Path
from src.data import load_dataset
from src.models import OptunaXGBoostClassifier
from sklearn.preprocessing import LabelEncoder

current_dir = Path(__file__).resolve().parent
cache_path = current_dir / '../data/param_cache'
n_trials = 500


def load_xgb_params(dataset_id):
    assert (cache_path.exists())
    if not (cache_path / f'{dataset_id}.json').exists():
        optimize_and_save(dataset_id)
    return load_cached_params(dataset_id)


def optimize_and_save(dataset_id):
    X, y = load_dataset(dataset_id)
    le = LabelEncoder()
    y = le.fit_transform(y)
    xgb = OptunaXGBoostClassifier(n_trials=n_trials)
    xgb.optimize(X, y)

    params, accuracy = xgb.get_study_results()

    result = {
        'params': params,
        'accuracy': accuracy,
        'n_trials': n_trials,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(cache_path / f'{dataset_id}.json', 'w') as f:
        json.dump(result, f)


def load_cached_params(dataset_id):
    assert (cache_path / f'{dataset_id}.json').exists()
    with open(cache_path / f'{dataset_id}.json', 'r') as f:
        return json.load(f)['params']


if __name__ == '__main__':

    print(
        load_xgb_params(17),
        load_xgb_params(19),
        load_xgb_params(43),
        load_xgb_params(52),
        load_xgb_params(94),
        load_xgb_params(96),
        load_xgb_params(151),
        load_xgb_params(174),
        load_xgb_params(176),
        load_xgb_params(267),
    )
