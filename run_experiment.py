import datetime
import json

from pathlib import Path
from functools import partial

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.experiment import ExperimentClassification
from src.lib import Settings, write_git_info
from src.models import tuned_decision_tree_classifier, OptunaXGBoostClassifier
from src.param_cache import parameter_lookup

experiment_slug = 'Test'


def main():
    print("Starting Experiment...")
    settings = Settings(
        n_repeats=5,
        n_splits=10,
        n_jobs=1,
    )

    output_path = Path(f'./output/results/{experiment_slug}') / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path.mkdir(parents=True)
    with open(output_path / 'settings.json', 'w') as f:
        json.dump(settings.__dict__, f)
    with open(output_path / 'info.txt', 'w') as f:
        write_git_info(f)

    #dataset_ids = [17, 43, 52, 94, 96, 151, 174, 176, 267]
    dataset_ids = [863]
    for dataset_id in dataset_ids:
        print(f'.... {dataset_id} ...')

        # This is not pretty ! But probably ok as a quick fix to add some kind of hyperparameter caching
        params = parameter_lookup(dataset_id)

        glass_box_base = partial(DecisionTreeClassifier, random_state=0, **params['decision_tree']['params'])
        glass_box_choices = {
            'decision_tree': glass_box_base,
        }

        black_box_base = partial(XGBClassifier, random_state=0, **params['xgboost']['params'])
        black_box_choices = {
            'xgboost': black_box_base,
        }

        glass_box_grader = partial(tuned_decision_tree_classifier, max_allowed_depth=4)
        grader_choices = {
            'decision_tree': glass_box_grader,
        }

        exp = ExperimentClassification(glass_box_choices, black_box_choices, grader_choices, settings)
        exp.run_experiment(dataset_id, output_path)
    print("Done!")


if __name__ == '__main__':
    main()
