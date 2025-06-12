from dataclasses import dataclass
import subprocess

import os
import socket
import numpy as np
import pandas as pd
import optuna

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


DIFFICULTY_EASY = 0
DIFFICULTY_HARD = 1
DIFFICULTY_VERY_HARD = 2
SENTINEL_REJECT = -13

def log_startup(logger, dataset_ids, settings):
    logger.info('Starting experiment')
    logger.info('Datasets: {}'.format(dataset_ids))
    logger.info('Settings: {}'.format(settings))
    result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE)
    logger.info('Git hash: {}'.format(result.stdout.decode('utf-8')))
    result = subprocess.run(['git', 'status', '.'], stdout=subprocess.PIPE)
    logger.info('Git status (./src): {}'.format(result.stdout.decode('utf-8')))
    logger.info(f'Running on system {socket.gethostname()}')

@dataclass
class Settings:
    n_repeats: int
    n_splits: int
    n_jobs: int = None
    save_X: bool = False
