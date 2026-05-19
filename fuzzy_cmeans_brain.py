"""
Fuzzy C-Means for Weighted Graphs with Modularity-Based Community Number Optimization
======================================================================================
Designed for functional brain connectivity matrices (e.g., rs-fMRI).

Pipeline:
    1. Accept a weighted symmetric connectivity matrix W (N x N)
    2. Find optimal number of communities C via modularity optimization
    3. Run Fuzzy C-Means on the graph using spectral initialization
    4. Return membership matrix U (N x C) — rows sum to 1 (valid probability distributions)
    5. Compute Community Participation Similarity Matrix S_JS using Jensen-Shannon divergence

Author: Generated for rs-fMRI brain network analysis
"""

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 1. MODULARITY MATRIX & OPTIMIZATION
# =============================================================================

def compute_modularity_matrix(W):
    """
    Compute the Newman-Girvan modularity matrix B for a weighted graph.

    B_ij = W_ij - (k_i * k_j) / (2m)

    Parameters
    ----------
    W : np.ndarray (N x N)
        Weighted symmetric adjacency matrix (e.g., FC matrix, non-negative).

    Returns
    -------
    B : np.ndarray (N x N)
        Modularity matrix.
    m : float
        Total edge weight (sum of all weights / 2).
    """
    k = W.sum(axis=1)           # weighted degree of each node
    m = k.sum() / 2.0           # total weight
    if m == 0:
        raise ValueError("Graph has no edges (all weights are zero).")
    B = W - np.outer(k, k) / (2 * m)
    return B, m


def compute_modularity_score(W, labels):
    """
    Compute modularity Q for a hard partition (used to evaluate each C).

    Q = (1/2m) * sum_{ij} [W_ij - k_i*k_j/(2m)] * delta(c_i, c_j)

    Parameters
    ----------
    W      : np.ndarray (N x N)  Weighted adjacency matrix.
    labels : np.ndarray (N,)     Hard community assignment.

    Returns
    -------
    Q : float   Modularity score in [-0.5, 1].
    """
    B, m = compute_modularity_matrix(W)
    N = len(labels)
    Q = 0.0
    for i in range(N):
        for j in range(N):
            if labels[i] == labels[j]:
                Q += B[i, j]
    Q /= (2 * m)
    return Q


def find_optimal_C(W, C_min=2, C_max=10, n_runs=5, m=2.0, random_state=42):
    """
    Find the optimal number of communities C by running FCM for each candidate
    C and selecting the one that maximizes modularity of the hard partition
    derived from the fuzzy memberships.

    Parameters
    ----------
    W            : np.ndarray (N x N)  Weighted connectivity matrix.
    C_min        : int                 Minimum number of communities to try.
    C_max        : int                 Maximum number of communities to try.
    n_runs       : int                 FCM runs per C (best of n_runs taken).
    m            : float               FCM fuzziness parameter (default 2.0).
    random_state : int                 Seed for reproducibility.

    Returns
    -------
    best_C   : int              Optimal number of communities.
    Q_scores : dict {C: Q}      Modularity score for each C tried.
    """
    rng = np.random.default_rng(random_state)
    Q_scores = {}

    print(f"Searching for optimal C in range [{C_min}, {C_max}]...")
    for C in range(C_min, C_max + 1):
        best_Q = -np.inf
        for run in range(n_runs):
            seed = int(rng.integers(0, 10000))
            U, _, _ = fuzzy_cmeans_graph(W, C=C, m=m, random_state=seed,
                                          verbose=False)
            # Hard partition: assign each node to its dominant community
            labels = np.argmax(U, axis=1)
            Q = compute_modularity_score(W, labels)
            if Q > best_Q:
                best_Q = Q
        Q_scores[C] = best_Q
        print(f"  C = {C:2d}  →  Q = {best_Q:.4f}")

    best_C = max(Q_scores, key=Q_scores.get)
    print(f"\n✓ Optimal C = {best_C}  (Q = {Q_scores[best_C]:.4f})\n")
    return best_C, Q_scores


# =============================================================================
# 2. SPECTRAL INITIALIZATION
# =============================================================================

def spectral_init(W, C, random_state=42):
    """
    Initialize FCM cluster centers using the top-C eigenvectors of the
    normalized Laplacian — more stable than random initialization for graphs.

    Parameters
    ----------
    W            : np.ndarray (N x N)  Weighted adjacency matrix.
    C            : int                 Number of communities.
    random_state : int

    Returns
    -------
    centers : np.ndarray (C x N)  Initial cluster centers in feature space.
    V       : np.ndarray (N x C)  Spectral embedding of nodes.
    """
    rng = np.random.default_rng(random_state)
    N = W.shape[0]

    # Normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
    d = W.sum(axis=1)
    d_safe = np.where(d > 0, d, 1.0)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d_safe))
    L_sym = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt

    # Smallest C eigenvectors (excluding the trivial zero eigenvalue)
    eigvals, eigvecs = eigh(L_sym)
    V = eigvecs[:, :C]  # (N x C) spectral embedding

    # Row-normalize for stability
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    V = V / norms

    # K-means++ style center selection in spectral space
    centers_idx = [int(rng.integers(0, N))]
    for _ in range(C - 1):
        dists = np.array([
            min(np.linalg.norm(V[i] - V[c]) ** 2 for c in centers_idx)
            for i in range(N)
        ])
        probs = dists / dists.sum()
        centers_idx.append(int(rng.choice(N, p=probs)))

    centers = V[centers_idx]  # (C x C) in spectral space
    return centers, V


# =============================================================================
# 3. FUZZY C-MEANS FOR WEIGHTED GRAPHS
# =============================================================================

def fuzzy_cmeans_graph(W, C, m=2.0, max_iter=300, tol=1e-6,
                        random_state=42, verbose=True):
    """
    Fuzzy C-Means adapted for weighted graphs.

    Uses the spectral embedding of the graph as the feature space,
    with the weighted connectivity matrix guiding distance computation.

    Membership update rule:
        u_ic = 1 / sum_k [ (d_ic / d_ik)^(2/(m-1)) ]

    where d_ic is the Euclidean distance from node i's spectral
    embedding to center c.

    Parameters
    ----------
    W            : np.ndarray (N x N)  Weighted symmetric adjacency matrix.
    C            : int                 Number of communities.
    m            : float               Fuzziness exponent (m > 1; typically 2).
    max_iter     : int                 Maximum iterations.
    tol          : float               Convergence threshold on U change.
    random_state : int
    verbose      : bool

    Returns
    -------
    U       : np.ndarray (N x C)  Membership matrix. Rows sum to 1.
    centers : np.ndarray (C x C)  Final cluster centers in spectral space.
    history : list of float        Objective function values per iteration.
    """
    N = W.shape[0]
    assert W.shape == (W.shape[0], W.shape[0]), "W must be square."
    assert m > 1, "Fuzziness parameter m must be > 1."

    # --- Spectral embedding ---
    centers, V = spectral_init(W, C, random_state=random_state)

    # --- Initialize membership matrix ---
    U = _init_membership(N, C, random_state)

    history = []
    exp = 2.0 / (m - 1)

    for iteration in range(max_iter):
        U_old = U.copy()

        # Update centers: c_k = sum_i (u_ik^m * v_i) / sum_i (u_ik^m)
        Um = U ** m  # (N x C)
        centers = (Um.T @ V) / (Um.sum(axis=0, keepdims=True).T + 1e-10)  # (C x C)

        # Compute distances from each node to each center
        dists = np.zeros((N, C))
        for k in range(C):
            diff = V - centers[k]
            dists[:, k] = np.linalg.norm(diff, axis=1)

        # Avoid division by zero: if a node is exactly on a center
        zero_mask = dists < 1e-10

        # Update memberships
        U = np.zeros((N, C))
        for i in range(N):
            if zero_mask[i].any():
                # Node exactly on a center: full membership to that center
                U[i, zero_mask[i]] = 1.0 / zero_mask[i].sum()
            else:
                for k in range(C):
                    ratio = dists[i, k] / (dists[i, :] + 1e-10)
                    U[i, k] = 1.0 / (ratio ** exp).sum()

        # Normalize rows to sum to 1
        row_sums = U.sum(axis=1, keepdims=True)
        U = U / np.where(row_sums > 0, row_sums, 1.0)

        # Objective function: J = sum_{i,k} u_ik^m * d_ik^2
        J = (Um * (dists ** 2)).sum()
        history.append(J)

        # Convergence check
        delta = np.linalg.norm(U - U_old)
        if verbose and (iteration % 20 == 0 or iteration < 5):
            print(f"  Iter {iteration:3d} | J = {J:.6f} | ΔU = {delta:.6f}")

        if delta < tol:
            if verbose:
                print(f"  Converged at iteration {iteration}.")
            break

    return U, centers, history


def _init_membership(N, C, random_state):
    """Random initialization of membership matrix with rows summing to 1."""
    rng = np.random.default_rng(random_state)
    U = rng.random((N, C))
    U = U / U.sum(axis=1, keepdims=True)
    return U


# =============================================================================
# 4. COMMUNITY PARTICIPATION SIMILARITY MATRIX (S_JS)
# =============================================================================

def compute_SJS(U):
    """
    Compute the Community Participation Similarity Matrix S_JS.

    S_JS[i, j] = 1 - sqrt(JS_divergence(u_i, u_j))

    where JS divergence is the Jensen-Shannon divergence between the
    membership vectors of nodes i and j.

    Since sqrt(JS) is a proper metric, (1 - sqrt(JS)) gives a similarity
    in [0, 1] where 1 = identical participation profiles.

    Parameters
    ----------
    U : np.ndarray (N x C)  Membership matrix from FCM (rows sum to 1).

    Returns
    -------
    S_JS : np.ndarray (N x N)  Symmetric similarity matrix in [0, 1].
    JS   : np.ndarray (N x N)  Raw JS divergence matrix in [0, 1].
    """
    N = U.shape[0]
    JS = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            # jensenshannon returns sqrt(JS divergence) in [0,1]
            js_dist = jensenshannon(U[i], U[j], base=2)
            JS[i, j] = js_dist ** 2   # JS divergence
            JS[j, i] = JS[i, j]

    S_JS = 1.0 - np.sqrt(JS)         # similarity: higher = more similar
    return S_JS, JS


# =============================================================================
# 5. FULL PIPELINE
# =============================================================================

def run_pipeline(W, C_min=2, C_max=10, m=2.0, n_runs=5,
                 random_state=42, verbose=True):
    """
    Full pipeline: Weighted FC matrix → Optimal C → FCM → U → S_JS

    Parameters
    ----------
    W            : np.ndarray (N x N)  Weighted functional connectivity matrix.
                   Should be non-negative and symmetric (apply threshold first).
    C_min        : int    Minimum communities to search.
    C_max        : int    Maximum communities to search.
    m            : float  FCM fuzziness (default 2.0).
    n_runs       : int    FCM runs per C for stability.
    random_state : int
    verbose      : bool

    Returns
    -------
    results : dict with keys:
        'U'       : np.ndarray (N x C)   Membership matrix
        'S_JS'    : np.ndarray (N x N)   Similarity matrix
        'JS'      : np.ndarray (N x N)   Raw JS divergence matrix
        'C'       : int                  Optimal number of communities
        'Q_scores': dict {C: Q}          Modularity scores per C
        'history' : list                 FCM objective history
    """
    N = W.shape[0]
    if verbose:
        print("=" * 60)
        print("Fuzzy C-Means for Weighted Brain Networks")
        print("=" * 60)
        print(f"Graph size: {N} nodes")
        print(f"Fuzziness m: {m}")
        print(f"Searching C in [{C_min}, {C_max}]\n")

    # Step 1: Find optimal C
    best_C, Q_scores = find_optimal_C(
        W, C_min=C_min, C_max=C_max,
        n_runs=n_runs, m=m, random_state=random_state
    )

    # Step 2: Run final FCM with best C (multiple runs, pick best)
    if verbose:
        print(f"Running final FCM with C = {best_C}...")
    rng = np.random.default_rng(random_state)
    best_U, best_centers, best_history = None, None, None
    best_J = np.inf

    for run in range(n_runs):
        seed = int(rng.integers(0, 10000))
        U, centers, history = fuzzy_cmeans_graph(
            W, C=best_C, m=m, random_state=seed, verbose=verbose
        )
        if history[-1] < best_J:
            best_J = history[-1]
            best_U = U
            best_centers = centers
            best_history = history

    # Step 3: Compute S_JS
    if verbose:
        print("\nComputing Community Participation Similarity Matrix (S_JS)...")
    S_JS, JS = compute_SJS(best_U)

    if verbose:
        print(f"\n✓ Done!")
        print(f"  Membership matrix U: shape {best_U.shape}")
        print(f"  S_JS matrix: shape {S_JS.shape}")
        print(f"  S_JS range: [{S_JS.min():.4f}, {S_JS.max():.4f}]")
        _print_community_summary(best_U, best_C)

    return {
        'U': best_U,
        'S_JS': S_JS,
        'JS': JS,
        'C': best_C,
        'Q_scores': Q_scores,
        'history': best_history,
    }


def _print_community_summary(U, C):
    """Print dominant community assignment and entropy for each node."""
    N = U.shape[0]
    labels = np.argmax(U, axis=1)
    entropy = -np.sum(U * np.log2(U + 1e-10), axis=1)
    max_entropy = np.log2(C)

    print(f"\n  Community summary (C = {C}):")
    for c in range(C):
        members = np.where(labels == c)[0]
        print(f"    Community {c+1}: {len(members)} dominant nodes")

    bridge_thresh = 0.7 * max_entropy
    bridge_nodes = np.where(entropy > bridge_thresh)[0]
    print(f"\n  Bridge nodes (entropy > 70% of max): {len(bridge_nodes)} nodes")
    if len(bridge_nodes) > 0 and len(bridge_nodes) <= 20:
        print(f"    Node indices: {bridge_nodes.tolist()}")


# =============================================================================
# 6. PREPROCESSING UTILITIES
# =============================================================================

def preprocess_fc_matrix(FC, threshold_pct=10, ensure_nonnegative=True):
    """
    Preprocess a functional connectivity matrix for graph-based analysis.

    Steps:
        1. Symmetrize (in case of floating point asymmetry)
        2. Zero the diagonal
        3. Optionally clip negative values to 0
        4. Apply proportional thresholding (keep top threshold_pct% of edges)

    Parameters
    ----------
    FC               : np.ndarray (N x N)  Raw Pearson correlation matrix.
    threshold_pct    : float   Percentage of strongest connections to keep (default 10%).
    ensure_nonnegative: bool   Clip negative correlations to 0 (default True).

    Returns
    -------
    W : np.ndarray (N x N)  Processed weighted adjacency matrix.
    """
    W = (FC + FC.T) / 2.0          # symmetrize
    np.fill_diagonal(W, 0)         # remove self-loops

    if ensure_nonnegative:
        W = np.clip(W, 0, None)    # FCM requires non-negative weights

    # Proportional threshold: keep top threshold_pct% of connections
    if threshold_pct < 100:
        upper = W[np.triu_indices_from(W, k=1)]
        cutoff = np.percentile(upper[upper > 0], 100 - threshold_pct)
        W[W < cutoff] = 0

    return W


# =============================================================================
# 7. DEMO / EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print("Generating synthetic brain-like connectivity matrix...")
    rng = np.random.default_rng(42)
    N = 50   # 50 brain regions (use 116 for AAL atlas)

    # Simulate block-structured FC matrix with overlapping regions
    true_C = 4
    block_size = N // true_C
    FC = rng.random((N, N)) * 0.1   # baseline noise
    FC = (FC + FC.T) / 2

    # Add community structure
    for c in range(true_C):
        start = c * block_size
        end = min(start + block_size, N)
        FC[start:end, start:end] += 0.6 + rng.random((end-start, end-start)) * 0.3

    # Add a few bridge nodes with connections to multiple communities
    for bridge in [block_size - 1, 2 * block_size - 1]:
        FC[bridge, :block_size*2] += 0.3
        FC[:block_size*2, bridge] += 0.3

    np.fill_diagonal(FC, 1.0)
    FC = np.clip(FC, -1, 1)

    # Preprocess
    print("\nPreprocessing FC matrix...")
    W = preprocess_fc_matrix(FC, threshold_pct=10)

    # Run full pipeline
    results = run_pipeline(W, C_min=2, C_max=7, m=2.0, n_runs=3,
                           random_state=42, verbose=True)

    U    = results['U']
    S_JS = results['S_JS']
    C    = results['C']
    Q_scores = results['Q_scores']

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0f1117')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    cmap_brain = plt.cm.RdYlBu_r
    cmap_mem   = plt.cm.viridis
    cmap_sim   = plt.cm.plasma

    def styled_ax(ax, title):
        ax.set_facecolor('#1a1d2e')
        ax.set_title(title, color='white', fontsize=11, pad=10, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        ax.tick_params(colors='#aaa', labelsize=8)
        return ax

    # 1. Preprocessed FC matrix
    ax1 = styled_ax(fig.add_subplot(gs[0, 0]), "Preprocessed FC Matrix (W)")
    im1 = ax1.imshow(W, cmap=cmap_brain, aspect='auto')
    plt.colorbar(im1, ax=ax1, fraction=0.046).ax.yaxis.set_tick_params(color='white', labelcolor='white')
    ax1.set_xlabel("Brain Region", color='#aaa', fontsize=8)
    ax1.set_ylabel("Brain Region", color='#aaa', fontsize=8)

    # 2. Modularity scores vs C
    ax2 = styled_ax(fig.add_subplot(gs[0, 1]), "Modularity Q vs. Number of Communities C")
    Cs = sorted(Q_scores.keys())
    Qs = [Q_scores[c] for c in Cs]
    ax2.plot(Cs, Qs, 'o-', color='#00d4ff', linewidth=2, markersize=8, markerfacecolor='white')
    ax2.axvline(x=C, color='#ff6b6b', linestyle='--', linewidth=1.5, label=f'Optimal C={C}')
    ax2.set_xlabel("Number of Communities (C)", color='#aaa', fontsize=9)
    ax2.set_ylabel("Modularity (Q)", color='#aaa', fontsize=9)
    ax2.legend(facecolor='#1a1d2e', edgecolor='#444', labelcolor='white', fontsize=9)

    # 3. Membership matrix U
    ax3 = styled_ax(fig.add_subplot(gs[0, 2]), f"Membership Matrix U (N×{C})")
    im3 = ax3.imshow(U, cmap=cmap_mem, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im3, ax=ax3, fraction=0.046).ax.yaxis.set_tick_params(color='white', labelcolor='white')
    ax3.set_xlabel("Community", color='#aaa', fontsize=8)
    ax3.set_ylabel("Brain Region", color='#aaa', fontsize=8)
    ax3.set_xticks(range(C))
    ax3.set_xticklabels([f"C{i+1}" for i in range(C)], color='#aaa', fontsize=8)

    # 4. S_JS matrix
    ax4 = styled_ax(fig.add_subplot(gs[1, 0]), "Community Participation Similarity (S_JS)")
    im4 = ax4.imshow(S_JS, cmap=cmap_sim, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im4, ax=ax4, fraction=0.046).ax.yaxis.set_tick_params(color='white', labelcolor='white')
    ax4.set_xlabel("Brain Region", color='#aaa', fontsize=8)
    ax4.set_ylabel("Brain Region", color='#aaa', fontsize=8)

    # 5. Membership entropy (bridge node detection)
    ax5 = styled_ax(fig.add_subplot(gs[1, 1]), "Node Membership Entropy\n(High = Bridge Node)")
    entropy = -np.sum(U * np.log2(U + 1e-10), axis=1)
    max_ent = np.log2(C)
    colors = ['#ff6b6b' if e > 0.7 * max_ent else '#00d4ff' for e in entropy]
    ax5.bar(range(N), entropy, color=colors, width=0.8, alpha=0.85)
    ax5.axhline(y=0.7 * max_ent, color='#ff6b6b', linestyle='--',
                linewidth=1.5, label='Bridge threshold (70% max entropy)')
    ax5.set_xlabel("Brain Region", color='#aaa', fontsize=8)
    ax5.set_ylabel("Shannon Entropy (bits)", color='#aaa', fontsize=8)
    ax5.legend(facecolor='#1a1d2e', edgecolor='#444', labelcolor='white', fontsize=8)

    # 6. FCM objective convergence
    ax6 = styled_ax(fig.add_subplot(gs[1, 2]), "FCM Objective Function Convergence")
    ax6.plot(results['history'], color='#a8ff78', linewidth=2)
    ax6.set_xlabel("Iteration", color='#aaa', fontsize=9)
    ax6.set_ylabel("Objective J", color='#aaa', fontsize=9)
    ax6.fill_between(range(len(results['history'])), results['history'],
                     alpha=0.15, color='#a8ff78')

    fig.suptitle(
        f"Fuzzy C-Means Brain Network Analysis  |  Optimal C = {C}  |  N = {N} regions",
        color='white', fontsize=14, fontweight='bold', y=0.98
    )

    out_path = "/mnt/user-data/outputs/fuzzy_cmeans_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nVisualization saved to: {out_path}")
    print("\nExample: How to use with your real fMRI data:")
    print("  import numpy as np")
    print("  from fuzzy_cmeans_brain import preprocess_fc_matrix, run_pipeline")
    print("  FC = np.load('your_fc_matrix.npy')   # shape (116, 116) for AAL")
    print("  W  = preprocess_fc_matrix(FC, threshold_pct=10)")
    print("  results = run_pipeline(W, C_min=2, C_max=10)")
    print("  U    = results['U']      # membership matrix (116 x C)")
    print("  S_JS = results['S_JS']   # similarity matrix (116 x 116)")
