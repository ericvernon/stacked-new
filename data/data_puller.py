import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from pathlib import Path

def download_uci(data_id):
    output_folder = Path('./uci') / str(data_id)
    if output_folder.exists():
        print(f'Data {data_id} already downloaded, skipping...')
        return

    output_folder.mkdir(exist_ok=True, parents=True)
    dataset = fetch_uci(data_id=data_id)
    save_dataset(dataset, output_folder)
    write_metadata(dataset, output_folder)


# There's probably some really clever Pythonic solution here... but for now...
def fetch_uci(data_id):
    if data_id == 19:
        return fetch_uci_19()

    if data_id == 225:
        return fetch_uci_225()

    if data_id == '242a':
        return fetch_uci_242('Y1')
    if data_id == '242b':
        return fetch_uci_242('Y2')

    return fetch_ucirepo(id=data_id)


def fetch_uci_19():
    dataset = fetch_ucirepo(id=19)
    oe = OrdinalEncoder(categories=[
        ['vhigh', 'high', 'med', 'low'],
        ['vhigh', 'high', 'med', 'low'],
        ['2', '3', '4', '5more'],
        ['2', '4', 'more'],
        ['small', 'med', 'big'],
        ['low', 'med', 'high'],
    ])
    dataset.data.features = pd.DataFrame(
        data=oe.fit_transform(dataset.data.features),
        columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    )
    return dataset


def fetch_uci_225():
    dataset = fetch_ucirepo(id=225)
    le = LabelEncoder()
    dataset.data.features.loc[:, 'Gender'] = le.fit_transform(dataset.data.features['Gender'])
    return dataset


def fetch_uci_242(idx):
    dataset = fetch_ucirepo(id=242)
    dataset.data.targets = dataset.data.targets[idx]
    return dataset


def save_dataset(dataset, output_folder):
    features = dataset.data.features
    targets = dataset.data.targets
    ds = features.join(targets)
    ds.to_csv(output_folder / 'data.txt', index=False)


def write_metadata(dataset, path):
    with open(path / 'info.txt', 'w', encoding='utf-8') as f:

        f.write(f"Dataset:     {dataset['metadata']['name']}\n")
        f.write(f"Abstract:    {dataset['metadata']['abstract']}\n")
        f.write(f"Authors:     {', '.join(dataset['metadata']['creators'])}\n")
        f.write(f"URL:         {dataset['metadata']['repository_url']}\n")
        f.write('\n')

        X = dataset.data.features.values
        y = dataset.data.targets.values.reshape(-1)
        le = LabelEncoder()
        y_bincount = np.bincount(le.fit_transform(y))

        n_samples = X.shape[0]
        n_features = X.shape[1]

        f.write(f'# Samples:   {n_samples}')
        if dataset['metadata']['has_missing_values'] == 'no':
            f.write(' (no missing values)\n')
        else:
            f.write(' (contains missing values)\n')
        f.write(f'# Features:  {n_features}\n')

        n_unique_y = len(y_bincount)
        original_labels = le.inverse_transform(range(n_unique_y))
        f.write(f'# Classes:   {n_unique_y}\n')

        for i in range(n_unique_y):
            n_bucket = y_bincount[i]
            n_bucket_pct = (100 * n_bucket) / n_samples
            f.write(f'{original_labels[i]:<12} {n_bucket:>6} ({n_bucket_pct:.2f}%)\n')


def main():
    datasets = [
        9,  # Auto MPG
        17,  # Breast Cancer Wisconsin (Diagnostic)
        19,  # Car Evaluation
        43,  # Haberman's Survival
        52,  # Ionosphere
        59,  # Letter Recognition
        60,  # Liver Disorders
        78,  # Page Blocks Classification
        94,  # spambase
        96,  # SPECTF Heart
        151,  # Connectionist Bench (Sonar, Mines vs. Rocks)
        159,  # MAGIC Gamma Telescope
        165,  # Concrete Compressive Strength
        172,  # Ozone Level Detection
        174,  # Parkinsons
        176,  # Blood Transfusion Service Center
        186,  # Wine quality
        212,  # Vertebral Column
        225,  # ILPD (Indian Liver Patient Dataset)
        '242a',  # Energy Efficiency
        '242b',  # Energy Efficiency
        267,  # Banknote Authentication
        291,  # Airfoil Self-Noise
        294,  # Combined Cycle Power Plant
        329,  # Diabetic Retinopathy Debrecen
        332,  # Online news popularity
        372,  # HTRU2
        451,  # Breast Cancer Coimbra
        464,  # Superconductivity Data
        477,  # Real Estate Valuation
        519,  # Heart Failure Clinical Records
        537,  # Cervical Cancer Behavior Risks
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
    for dataset in datasets:
        download_uci(dataset)


if __name__ == '__main__':
    main()