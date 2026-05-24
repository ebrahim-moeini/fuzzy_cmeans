import os
from fuzzy_cmeans_brain import group_analysis
import pickle
import glob

def get_files_and_labels(data_root, algorithm_name):
    groups = {'HC': [], 'EMCI': [], 'LMCI': [], 'AD': []}
    for grp in groups:
        groups[grp] = sorted(glob.glob(
            os.path.join(data_root, grp, f"{algorithm_name}-*.pkl")
        ))
    filepaths, labels = [], []
    for grp, fps in groups.items():
        filepaths.extend(fps)
        labels.extend([grp] * len(fps))
    return filepaths, labels

def main():
    algorithm_name = "fcm"
    dataset_sub_dir = 'fc_groups'

    # Get the current directory and construct paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_results_dir = os.path.join(os.path.dirname(current_dir), f'results/{dataset_sub_dir}')

    filepaths, labels = get_files_and_labels(base_results_dir, algorithm_name)

    SJS_list = []
    for fp in filepaths:
        with open(fp, 'rb') as file:
            res = pickle.load(file)
        SJS_list.append(res['S_JS'])

    stats = group_analysis(SJS_list, labels, alpha=0.05)
    # stats['sig_pairs'] → list of (roi_i, roi_j, F, p_fdr)

    with open(os.path.join(base_results_dir, f'{algorithm_name}_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)

if __name__ == "__main__":
    main()
