import numpy as np
from fuzzy_cmeans_brain import visualize_results, run_pipeline
import pickle
import os

# =============================================================================
# 12.  EXAMPLE USAGE
# =============================================================================
def run(data_set:dict, datasets_dir:str, results_dir:str):

    # ── Single-subject ──────────────────────────────────────────────────
    results = run_pipeline(
        fc_path_or_matrix= os.path.join(datasets_dir, data_set["dataset"]),
        C_min             = 2,
        C_max             = 10,
        m                 = 2.0,        # standard fuzziness
        n_runs            = 5,          # restarts for stability
        threshold_pct     = 10,         # keep top 10% edges
        negative_strategy = 'zero',     # recommended for Pearson FC
        fisher_z          = True,       # recommended for Pearson r
        random_state      = 42,
        verbose           = True,
        node_number       = 162,
    )

    U       = results['U']          # (162, C) — membership matrix
    S_JS    = results['S_JS']       # (162, 162) — similarity matrix
    C_opt   = results['C']          # optimal number of communities
    bridges = results['bridge_info']

    # Save
    subject =  data_set["dataset"].removeprefix("fc_").removesuffix(".csv")
    U_name = f"U_{subject}"
    # np.save("U_002_S_0295.npy",   U)
    np.save(os.path.join(results_dir, f'{data_set["algorithm"]}-{U_name}.npy'), U)
    # np.save("SJS_002_S_0295.npy", S_JS)
    S_JS_name = f"SJS_{subject}"
    np.save(os.path.join(results_dir, f'{data_set["algorithm"]}-{S_JS_name}.npy'), S_JS)

    # with open('002_S_0295.pkl', 'wb') as f:
    with open(os.path.join(results_dir, f'{data_set["algorithm"]}-{subject}.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Visualise
    visualize_results(
        results,
        subject_id = subject,
        save_path=os.path.join(results_dir, f'{data_set["algorithm"]}-{subject}.png'),
        # save_path  = "results_002_S_0295.png",
    )

    return results

    # ── Multi-subject  (uncomment when you have the full cohort) ────────
    #
    # import glob, os
    #
    # def get_files_and_labels(data_root):
    #     groups = {'HC': [], 'EMCI': [], 'LMCI': [], 'AD': []}
    #     for grp in groups:
    #         groups[grp] = sorted(glob.glob(
    #             os.path.join(data_root, grp, "fc_*.csv")
    #         ))
    #     filepaths, labels = [], []
    #     for grp, fps in groups.items():
    #         filepaths.extend(fps)
    #         labels.extend([grp] * len(fps))
    #     return filepaths, labels
    #
    # filepaths, labels = get_files_and_labels("data/")
    # SJS_list = []
    # for fp in filepaths:
    #     res = run_pipeline(fp, C_min=2, C_max=10, verbose=False)
    #     SJS_list.append(res['S_JS'])
    #
    # stats = group_analysis(SJS_list, labels, alpha=0.05)
    # # stats['sig_pairs'] → list of (roi_i, roi_j, F, p_fdr)
