import os
from pathlib import Path
from run import run
from fuzzy_cmeans_brain import group_analysis
import pickle

def main():
    """
    Main function to run the Louvain algorithm

    Returns:
    --------
    None

    """

    algorithm_name = "fcm"
    groups = ["AD", "CN", "EMCI", "LMCI"]
    run_all = True

    dataset_sub_dir = 'fc_groups'

    data_set = {
        "nodes_number": 162,
        "algorithm": algorithm_name,
    }

    # Get the current directory and construct paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(os.path.dirname(current_dir), f'datasets/{dataset_sub_dir}')
    base_results_dir = os.path.join(os.path.dirname(current_dir), f'results/{dataset_sub_dir}')

    Path(base_results_dir).mkdir(parents=True, exist_ok=True)

    if not run_all:
        data_set["dataset"] = "fc_002_S_0295_2012-05-10.csv"
        result = run(data_set, datasets_dir, base_results_dir)
    # run one dataset
    else:
        for group in groups:
            datasets_dir = os.path.join(os.path.dirname(current_dir), f'datasets/{dataset_sub_dir}/{group}')
            results_dir = os.path.join(os.path.dirname(current_dir), f'results/{dataset_sub_dir}/{group}')
            # run all datasets
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            data_set["nodes_number"] = 162
            for subject in os.listdir(os.path.join(datasets_dir)):
                if subject.endswith(".csv"):
                    print(f"subject {subject} is being processed")
                    print("===============================================================")
                    data_set["dataset"] = subject
                    result = run(data_set, datasets_dir, results_dir)


if __name__ == "__main__":
    main()