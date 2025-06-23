import datetime
from pathlib import Path

import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier

from src.data import load_dataset
from src.experiment import ExperimentClassification
from src.lib import Settings
from src.models import *


def main():
    print("Starting mini-experiment...")
    settings = Settings(
        n_repeats=1,
        n_splits=4,
        n_jobs=1,
    )

    glass_box_choices = {
        'ShallowDecisionTree': shallow_decision_tree_classifier,
    }

    black_box_choices = {
        'XGBoost': XGBClassifier,
    }

    grader_choices = {
        'ShallowDecisionTree': shallow_decision_tree_classifier,
        'XGBoost': XGBClassifier,
    }

    exp = ExperimentClassification(glass_box_choices, black_box_choices, grader_choices, settings)
    output_path = Path('./output/results/simple_experiment') / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path.mkdir(parents=True)

    dataset_id = 94
    exp.run_experiment(dataset_id, output_path)
    print("Done!")

    pass


if __name__ == '__main__':
    main()
