import json

from sklearn.preprocessing import LabelEncoder

from collections import deque
from sklearn.model_selection import RepeatedStratifiedKFold

from functools import partial
from src.models import tuned_decision_tree_classifier, OptunaXGBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import clone

from src.data import load_dataset


def main():
    datasets = [
            17,  # Breast Cancer Wisconsin (Diagnostic)
            19,  # Car Evaluation
            43,  # Haberman's Survival
            52,  # Ionosphere
            59,  # Letter Recognition
            78,  # Page Blocks Classification
            94,  # Spambase
            96,  # SPECTF Heart
            151,  # Connectionist Bench (Sonar, Mines vs. Rocks)
            159,  # MAGIC Gamma Telescope
            174,  # Parkinsons
            176,  # Blood Transfusion Service Center
            212,  # Vertebral Column
            267,  # Banknote authentication
            329,  # Diabetic Retinopathy Debrecen
            372,  # HTRU2
            451,  # Breast Cancer Coimbra
            519,  # Heart Failure Clinical Records
            537,  # Cervical Cancer Behavior Risk
            545,  # Rice (Cammeo and Osmancik)
            563,  # Iranian Churn
            572,  # Taiwanese Bankruptcy Prediction
            602,  # Dry Bean
            722,  # NATICUSdroid (Android Permissions)
            827,  # Sepsis Survival Minimal Clinical Records
            850,  # Raisin
            863,  # Maternal health risk
            887,  # National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset
            890,  # AIDS Clinical Trials Group Study 175
            891,  # CDC Diabetes Health Indicators
    ]

    sizes = []
    for d_id in datasets:
        X, y = load_dataset(d_id)
        sizes.append((d_id, len(X)))

    sizes.sort(key=lambda x: x[1])

    with open("collate_complexity_results.txt", "w") as fh:
        for size in sizes:
            fh.write(f"ID {size[0]} (N = {size[1]})\n")

        smallest = sizes[0]
        median = sizes[len(sizes) // 2]
        largest = sizes[-1]

        fh.write(f"Smallest Dataset: ID {smallest[0]} (N = {smallest[1]})\n")
        fh.write(f"Median Dataset:   ID {median[0]} (N = {median[1]})\n")
        fh.write(f"Largest Dataset:  ID {largest[0]} (N = {largest[1]})\n")

        fh.write(str(len(sizes)) + "\n")

    with open("n_steps.json", "w") as fh:
        n_steps_many_passes_full = []
        many_pass_basket_full = [94, 545, 563,
                            887, 890, 19,
                            267, 329, 863,
                            850, 176, 17,
                            52, 212, 43,
                            519, 96, 151,
                            174, 451, 537,
        ]

        for dataset_id in many_pass_basket_full:
            print(f'dataset_id: {dataset_id}')
            X, y = load_dataset(dataset_id)
            le = LabelEncoder()
            y = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

            gb_model = tuned_decision_tree_classifier(max_allowed_depth=4, n_jobs=-1)
            gb_model.fit(X_train, y_train)

            bb_model = OptunaXGBoostClassifier(n_trials=100, n_jobs=-1)
            bb_model.fit(X_train, y_train)

            dt_estimator = gb_model.best_estimator_
            gb_model_fn = partial(clone, dt_estimator)

            frozen_params = bb_model.get_study_results()[0]
            bb_model_fn = partial(bb_model._model_fn, random_state=0, **frozen_params)

            glass_results = count_calibration_steps_optimized(gb_model_fn, X_train=X_train, y_train=y_train, n_splits=5, max_stop=10)
            black_results = count_calibration_steps_optimized(bb_model_fn, X_train=X_train, y_train=y_train, n_splits=5, max_stop=10)

            for i in range(10):
                n_passes = i + 1
                n_steps_many_passes_full.append({
                    'dataset_id': dataset_id,
                    'n_passes': n_passes,
                    'n_steps_glass': glass_results[i],
                    'n_steps_black': black_results[i],
                })

        json_string = json.dumps(n_steps_many_passes_full, indent=4)
        fh.write(json_string)


def count_calibration_steps_optimized(model_function, X_train, y_train, n_splits, max_stop):
    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=555_555, random_state=0)

    all_incorrect_idx = set()
    total_cv_splits = 0

    consecutive_stable_passes = 0
    new_in_current_pass = 0

    # results[i] will store the splits needed to reach (i + 1) stable passes
    results = [None] * max_stop

    for data_split_idx, (train_idx, calibration_idx) in enumerate(kfold.split(X_train, y_train)):
        total_cv_splits += 1
        model = model_function()
        model.fit(X_train[train_idx], y_train[train_idx])

        wrong_idx_within_calibration = model.predict(X_train[calibration_idx]) != y_train[calibration_idx]
        wrong_idx_within_training = calibration_idx[wrong_idx_within_calibration]

        n_before = len(all_incorrect_idx)
        all_incorrect_idx.update(wrong_idx_within_training)
        n_new = len(all_incorrect_idx) - n_before

        new_in_current_pass += n_new

        # Evaluate only at the end of a full CV pass (e.g., every 5 splits)
        if total_cv_splits % n_splits == 0:
            if new_in_current_pass == 0:
                consecutive_stable_passes += 1
            else:
                consecutive_stable_passes = 0

            # Check if we just crossed any new stability thresholds
            for i in range(1, max_stop + 1):
                # If we achieved 'i' stable passes and haven't recorded it yet
                if consecutive_stable_passes >= i and results[i-1] is None:
                    results[i-1] = total_cv_splits

            # Reset the counter for the next CV pass
            new_in_current_pass = 0

            # If the array is fully populated, we've found all stopping conditions!
            if all(r is not None for r in results):
                break

    return results


if __name__ == '__main__':
    main()
