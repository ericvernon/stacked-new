from dataclasses import dataclass
from pathlib import Path

reports_root = Path('./output/reports')
results_root = Path('./output/results')


def main():
    experiment_name = 'simple_experiment/20250619-175829'

    reports_path = reports_root / experiment_name
    reports_path.mkdir(exist_ok=True, parents=True)

    raw_data = {}
    results_path = results_root / experiment_name
    for dataset in results_path.iterdir():
        assert dataset.is_dir()
        for results_file in dataset.iterdir():
            run_info = parse_filename(results_file.name)
            algorithm_type = run_info.algorithm_type_hash()
            if algorithm_type not in raw_data:
                raw_data[algorithm_type] = {}
            if dataset.name not in raw_data[algorithm_type]:
                raw_data[algorithm_type][dataset.name] = {
                    'train': {
                        'glass_n_correct_easy': 0,
                        'glass_n_correct_hard': 0,
                        'black_n_correct_easy': 0,
                        'black_n_correct_hard': 0,
                        'hybrid_n_correct_easy': 0,
                        'hybrid_n_correct_hard': 0,
                    },
                    'test': {
                        'glass_n_correct_easy': 0,
                        'glass_n_correct_hard': 0,
                        'black_n_correct_easy': 0,
                        'black_n_correct_hard': 0,
                        'hybrid_n_correct_easy': 0,
                        'hybrid_n_correct_hard': 0,
                    },
                }
            print(algorithm_type)
#        raw_data[dataset.name] = dataset_results
    print(raw_data)


@dataclass
class RunInfo:
    run_id: int
    glass_box_type: str
    black_box_type: str
    grader_type: str
    train_or_test: str
    grader_degree: str

    def algorithm_type_hash(self):
        return f'{self.glass_box_type}-{self.black_box_type}-{self.grader_type}-{self.grader_degree}'


def parse_filename(filename):
    bits = filename[:-4].split('-')
    print(bits)
    return RunInfo(
        run_id=int(bits[0]),
        glass_box_type=bits[1],
        black_box_type=bits[2],
        grader_type=bits[3],
        train_or_test=bits[4],
        grader_degree=bits[5],
    )


if __name__ == '__main__':
    main()
    pass