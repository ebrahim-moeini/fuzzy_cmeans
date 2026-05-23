import numpy as np
from fuzzy_cmeans_brain import visualize_results, run_pipeline

# =============================================================================
# 12.  EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":

    # ── Single-subject ──────────────────────────────────────────────────
    results = run_pipeline(
        fc_path_or_matrix = "fc_002_S_0295_2012-05-10.csv",
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
    np.save("U_002_S_0295.npy",   U)
    np.save("SJS_002_S_0295.npy", S_JS)

    # Visualise
    visualize_results(
        results,
        subject_id = "002_S_0295",
        save_path  = "results_002_S_0295.png",
    )

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
