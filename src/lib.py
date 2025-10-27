from dataclasses import dataclass
import subprocess

import json
import socket

DIFFICULTY_EASY = 0
DIFFICULTY_HARD = 1
DIFFICULTY_VERY_HARD = 2
SENTINEL_REJECT = -13


def write_git_info(fh):
    result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE)
    fh.write('Git hash: {}\n'.format(result.stdout.decode('utf-8')))

    result = subprocess.run(['git', 'status', '.'], stdout=subprocess.PIPE)
    fh.write('Git status (./src): {}\n\n'.format(result.stdout.decode('utf-8')))

    fh.write(f'Running on system {socket.gethostname()}\n')


@dataclass
class Settings:
    n_repeats: int
    n_splits: int
    n_jobs: int = None
    cw_n_splits: int = None
    save_X: bool = False
