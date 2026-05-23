"""
Fuzzy C-Means for Weighted Brain Functional Connectivity Networks
=================================================================
Tailored specifically for ADNI rs-fMRI FC matrices with the format:
  - CSV: first column = ROI index labels (integers, non-consecutive)
  - 162 x 162 symmetric Pearson correlation matrix
  - Region labels 1–169 with 7 missing ROIs (35,36,81,82,133,134,168)
  - Diagonal = 1.0 (self-correlation), no NaN values
  - Negative correlations present (range ~ -0.92 to 1.0)

Full Pipeline:
    1.  Load FC matrix from CSV  →  extract ROI labels & data matrix
    2.  Preprocess               →  Fisher-z, zero diagonal, handle negatives,
                                    proportional threshold
    3.  Find optimal C           →  modularity optimization over C_min..C_max
    4.  Fuzzy C-Means            →  spectral initialization + FCM iterations
    5.  Membership matrix U      →  (N x C), rows sum to 1
    6.  S_JS matrix              →  pairwise Jensen-Shannon similarity
    7.  Bridge node detection    →  membership entropy thresholding
    8.  Group-level ANOVA        →  FDR-corrected comparison across HC/MCI/AD
    9.  Visualization            →  6-panel diagnostic figure

Dependencies:
    pip install numpy scipy pandas matplotlib statsmodels

Author: Generated for ADNI rs-fMRI Alzheimer's Disease research
"""

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 1.  DATA LOADING
# =============================================================================

def load_fc_matrix(filepath, node_number=None):
    """
    Load one subject's FC matrix from an ADNI-format CSV file.

    CSV format (confirmed from dataset inspection):
      - Row 0 / Col 0: integer ROI index (e.g., 1..169 with gaps)
      - Remaining 162 x 162 values: Pearson r, symmetric, diagonal = 1.0
      - Non-consecutive column labels due to 7 excluded ROIs:
        35, 36, 81, 82, 133, 134, 168

    Parameters
    ----------
    node_number : ٔNumber of expected node that should be in the FC matrix
    filepath : str
        Path to the subject CSV file,
        e.g. 'fc_002_S_0295_2012-05-10.csv'

    Returns
    -------
    FC         : np.ndarray (162, 162)  Raw Pearson correlation matrix.
    roi_labels : list[int]              ROI integer labels in original order,
                                        e.g. [1,2,...,34,37,...,169].
    """
    df = pd.read_csv(filepath, index_col=0)

    # Cast index & column labels to int (they come in as strings/floats)
    df.index   = df.index.astype(int)
    df.columns = df.columns.astype(int)

    # Sanity check: index and columns must agree
    assert list(df.index) == list(df.columns), (
        "Row and column ROI labels do not match — check the CSV format."
    )

    roi_labels = list(df.index)          # e.g. [1,2,...,34,37,...,169]
    FC = df.values.astype(np.float64)    # (162, 162)

    if node_number is None:
        node_number = FC.shape[0]

    assert FC.shape == (node_number, node_number), f"Expected {node_number, node_number} square matrix, but it shape is {FC.shape}."

    # Symmetrize in case of tiny floating-point asymmetry
    if not np.allclose(FC, FC.T, atol=1e-5):
        print("[load] Warning: matrix not perfectly symmetric — symmetrizing.")
        FC = (FC + FC.T) / 2.0

    print(f"[load] {node_number} regions | labels {roi_labels[0]}..{roi_labels[-1]} "
          f"(7 ROIs excluded) | "
          f"r ∈ [{FC[~np.eye(node_number, dtype=bool)].min():.3f}, "
          f"{FC[~np.eye(node_number, dtype=bool)].max():.3f}]")

    return FC, roi_labels


def load_subjects(filepaths, node_number):
    """
    Load FC matrices for a list of subjects.

    Parameters
    ----------
    filepaths : list[str]   One CSV path per subject.
    node_number : ٔNumber of expected node that should be in the FC matrix

    Returns
    -------
    FC_list    : list[np.ndarray]   One (162,162) matrix per subject.
    roi_labels : list[int]          ROI labels (consistent across subjects).
    """
    FC_list, roi_labels = [], None
    for fp in filepaths:
        FC, labels = load_fc_matrix(fp, node_number)
        FC_list.append(FC)
        if roi_labels is None:
            roi_labels = labels
    print(f"[load] Loaded {len(FC_list)} subjects.\n")
    return FC_list, roi_labels


# =============================================================================
# 2.  PREPROCESSING
# =============================================================================

def preprocess_fc_matrix(FC, threshold_pct=10,
                          negative_strategy='zero',
                          fisher_z=True):
    """
    Prepare a raw Pearson FC matrix for weighted graph analysis.

    Steps
    -----
    1. **Fisher r-to-z transform** (recommended for Pearson r):
       z = arctanh(r).  Stabilises variance across the [-1,1] range.
       Diagonal is zeroed before the transform to avoid arctanh(1) = ∞.

    2. **Zero the diagonal** — self-connections are not meaningful
       for community detection.

    3. **Handle negative correlations** via `negative_strategy`:
       - ``'zero'``     : clip negatives to 0.
                          *Recommended* — FCM and modularity assume
                          non-negative weights; negative FC edges have
                          ambiguous community meaning.
       - ``'absolute'`` : |r| — preserves anti-correlation magnitude.
       - ``'keep'``     : leave negatives as-is (only valid if your
                          downstream algorithm supports signed graphs).

    4. **Proportional thresholding**: retain the top `threshold_pct`%
       of positive edge weights (computed on the upper triangle).
       Weaker edges are set to 0.  This sparsifies the graph while
       preserving the strongest functional connections.

    5. **Re-symmetrize** after thresholding.

    Parameters
    ----------
    FC                : np.ndarray (N, N)
        Raw Pearson correlation matrix (diagonal = 1.0).
    threshold_pct     : float
        Percentage of strongest positive connections to keep (default 10).
    negative_strategy : {'zero', 'absolute', 'keep'}
        How to handle negative correlations.
    fisher_z          : bool
        Apply Fisher r-to-z transform (default True).

    Returns
    -------
    W        : np.ndarray (N, N)  Processed weighted adjacency matrix.
    cutoff   : float              Threshold value applied to edge weights.
    """
    W = FC.copy()
    N = W.shape[0]

    # --- Step 1: Fisher r-to-z ---
    if fisher_z:
        np.fill_diagonal(W, 0)
        W = np.clip(W, -0.9999, 0.9999)   # guard against ±1 on diagonal
        W = np.arctanh(W)

    # --- Step 2: Zero diagonal ---
    np.fill_diagonal(W, 0)

    # --- Step 3: Handle negatives ---
    if negative_strategy == 'zero':
        W = np.clip(W, 0, None)
    elif negative_strategy == 'absolute':
        W = np.abs(W)
    elif negative_strategy == 'keep':
        pass
    else:
        raise ValueError(
            f"Unknown negative_strategy '{negative_strategy}'. "
            "Choose 'zero', 'absolute', or 'keep'."
        )

    # --- Step 4: Proportional threshold on upper triangle ---
    upper_vals = W[np.triu_indices(N, k=1)]
    pos_vals   = upper_vals[upper_vals > 0]

    if len(pos_vals) == 0:
        raise ValueError(
            "No positive connections remain after preprocessing. "
            "Try 'absolute' instead of 'zero' for negative_strategy, "
            "or reduce threshold_pct."
        )

    cutoff = np.percentile(pos_vals, 100 - threshold_pct)
    W[W < cutoff] = 0

    # --- Step 5: Re-symmetrize ---
    W = (W + W.T) / 2.0

    n_edges = int((W > 0).sum() // 2)
    density = n_edges / (N * (N - 1) / 2)
    print(f"[preprocess] strategy='{negative_strategy}' | "
          f"fisher_z={fisher_z} | threshold={threshold_pct}%")
    print(f"[preprocess] cutoff={cutoff:.4f} | "
          f"edges={n_edges} | density={density:.3f} | "
          f"w ∈ [{W[W>0].min():.4f}, {W.max():.4f}]")

    return W, cutoff


# =============================================================================
# 3.  MODULARITY & OPTIMAL C
# =============================================================================

def _modularity_matrix(W):
    """
    Newman-Girvan modularity matrix  B_ij = W_ij - k_i*k_j / (2m).

    Returns B (N,N) and m (total edge weight).
    """
    k = W.sum(axis=1)
    m = k.sum() / 2.0
    if m == 0:
        raise ValueError("Graph has no edges — check preprocessing.")
    return W - np.outer(k, k) / (2.0 * m), m


def compute_modularity(W, labels):
    """
    Modularity Q for a hard-partition vector `labels`.

        Q = (1/2m) Σ_ij B_ij δ(c_i, c_j)

    Parameters
    ----------
    W      : np.ndarray (N, N)  Weighted adjacency matrix.
    labels : np.ndarray (N,)    Integer community label per node.

    Returns
    -------
    Q : float   Modularity in [-0.5, 1].  Higher = better partition.
    """
    B, m = _modularity_matrix(W)
    N    = len(labels)
    Q    = 0.0
    for i in range(N):
        for j in range(N):
            if labels[i] == labels[j]:
                Q += B[i, j]
    return Q / (2.0 * m)


def find_optimal_C(W, C_min=2, C_max=10, n_runs=5,
                   m_fuzzy=2.0, random_state=42):
    """
    Search for the optimal number of communities C by modularity.

    For each C in [C_min, C_max]:
      1. Run FCM `n_runs` times (different seeds).
      2. Convert best U to hard labels via argmax.
      3. Compute Q.
    Return the C that maximises Q.

    Parameters
    ----------
    W            : np.ndarray (N, N)  Preprocessed adjacency matrix.
    C_min        : int                Lower bound of search.
    C_max        : int                Upper bound of search.
    n_runs       : int                FCM restarts per C.
    m_fuzzy      : float              FCM fuzziness exponent.
    random_state : int                Base random seed.

    Returns
    -------
    best_C   : int          Optimal C.
    Q_scores : dict[int,float]  {C: Q} for every C tried.
    """
    rng = np.random.default_rng(random_state)
    Q_scores = {}

    print(f"\n[find_C] Searching C ∈ [{C_min}, {C_max}]  "
          f"({n_runs} runs each) …")
    print(f"{'C':>4}  {'Q':>10}")
    print("─" * 18)

    for C in range(C_min, C_max + 1):
        best_Q = -np.inf
        for _ in range(n_runs):
            seed = int(rng.integers(0, 100_000))
            U, _, _ = _fcm(W, C=C, m=m_fuzzy,
                           random_state=seed, verbose=False)
            Q = compute_modularity(W, np.argmax(U, axis=1))
            if Q > best_Q:
                best_Q = Q
        Q_scores[C] = best_Q
        flag = "  ← best" if best_Q == max(Q_scores.values()) else ""
        print(f"{C:>4}  {best_Q:>10.5f}{flag}")

    best_C = max(Q_scores, key=Q_scores.get)
    print(f"\n[find_C] Optimal C = {best_C}  "
          f"(Q = {Q_scores[best_C]:.5f})\n")
    return best_C, Q_scores


# =============================================================================
# 4.  SPECTRAL INITIALISATION
# =============================================================================

def _spectral_embedding(W, C):
    """
    Compute the C-dimensional spectral embedding of the graph.

    Uses the C smallest eigenvectors of the normalised symmetric
    Laplacian  L = I - D^{-1/2} W D^{-1/2}.

    Row-normalises the result so that FCM distances are scale-invariant.

    Returns
    -------
    V : np.ndarray (N, C)   One row per node (spectral coordinates).
    """
    N  = W.shape[0]
    d  = W.sum(axis=1)
    ds = np.where(d > 0, d, 1.0)
    Di = np.diag(1.0 / np.sqrt(ds))
    L  = np.eye(N) - Di @ W @ Di

    # Request C+1 eigenvectors; the smallest (≈ 0) is the trivial one
    try:
        _, vecs = eigh(L, subset_by_index=[0, min(C, N - 1)])
        V = vecs[:, :C]
    except Exception:
        _, vecs = eigh(L)
        V = vecs[:, :C]

    # Row-normalise
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.where(norms > 1e-10, norms, 1.0)
    return V


def _kmeans_pp_centers(V, C, rng):
    """
    k-means++ centre selection in spectral space V  (N, C).

    Returns centers_idx : list[int]  — indices of chosen centre nodes.
    """
    N = V.shape[0]
    idx = [int(rng.integers(0, N))]
    for _ in range(C - 1):
        dists = np.array([
            min(np.linalg.norm(V[i] - V[c]) ** 2 for c in idx)
            for i in range(N)
        ])
        total = dists.sum()
        probs = dists / total if total > 0 else np.ones(N) / N
        idx.append(int(rng.choice(N, p=probs)))
    return idx


# =============================================================================
# 5.  FUZZY C-MEANS (core)
# =============================================================================

def _fcm(W, C, m=2.0, max_iter=500, tol=1e-7,
         random_state=42, verbose=True):
    """
    Fuzzy C-Means on a weighted graph via spectral embedding.

    Feature space: N-node graph → (N, C) spectral coordinates V.

    Membership update  (standard FCM):
        u_ic = (d_ic^{-1/(m-1)}) / Σ_k d_ik^{-1/(m-1)}

    Centre update:
        c_k = Σ_i u_ik^m v_i / Σ_i u_ik^m

    Objective (minimised):
        J = Σ_i Σ_k u_ik^m ||v_i - c_k||²

    Parameters
    ----------
    W            : np.ndarray (N, N)  Non-negative weighted adjacency matrix.
    C            : int                Number of communities.
    m            : float              Fuzziness exponent (>1; default 2.0).
    max_iter     : int                Maximum iterations.
    tol          : float              Convergence criterion on ||ΔU||.
    random_state : int
    verbose      : bool

    Returns
    -------
    U       : np.ndarray (N, C)   Membership matrix; rows sum to 1.
    centers : np.ndarray (C, C)   Final cluster centres in spectral space.
    history : list[float]          Objective J per iteration.
    """
    assert m > 1.0, "Fuzziness m must be > 1."
    assert C >= 2,  "C must be >= 2."

    N   = W.shape[0]
    rng = np.random.default_rng(random_state)
    exp = 1.0 / (m - 1.0)          # exponent for membership formula

    # --- Spectral embedding ---
    V = _spectral_embedding(W, C)   # (N, C)

    # --- k-means++ initialisation ---
    c_idx   = _kmeans_pp_centers(V, C, rng)
    centers = V[c_idx].copy()       # (C, C)

    # --- Random initial U (will be overwritten in iteration 0 center update) ---
    U = rng.random((N, C))
    U = U / U.sum(axis=1, keepdims=True)

    history = []

    for it in range(max_iter):
        U_prev = U.copy()

        # ── Centre update ──────────────────────────────────────────────
        Um         = U ** m                                 # (N, C)
        w_sum      = Um.sum(axis=0)                         # (C,)
        w_sum      = np.where(w_sum > 1e-12, w_sum, 1e-12)
        centers    = (Um.T @ V) / w_sum[:, np.newaxis]     # (C, C)

        # ── Distance: d[i,k] = ||v_i - c_k||² ─────────────────────────
        D = np.zeros((N, C))
        for k in range(C):
            diff    = V - centers[k]                        # (N, C)
            D[:, k] = (diff * diff).sum(axis=1)

        D = np.maximum(D, 1e-12)                            # numerical floor

        # ── Membership update ───────────────────────────────────────────
        # u_ic = D[i,c]^{-exp} / Σ_k D[i,k]^{-exp}
        inv_D  = D ** (-exp)                                # (N, C)
        r_sums = inv_D.sum(axis=1, keepdims=True)
        U      = inv_D / np.where(r_sums > 0, r_sums, 1.0)

        # ── Objective ───────────────────────────────────────────────────
        J = float((U ** m * D).sum())
        history.append(J)

        delta = float(np.linalg.norm(U - U_prev))
        if verbose and (it < 3 or it % 50 == 0):
            print(f"  iter {it:4d} | J={J:14.4f} | ΔU={delta:.2e}")

        if delta < tol:
            if verbose:
                print(f"  Converged at iter {it+1}  (ΔU={delta:.2e})")
            break
    else:
        if verbose:
            print(f"  [warn] max_iter={max_iter} reached (ΔU={delta:.2e})")

    # Final row-normalise (cleanup floating-point drift)
    r = U.sum(axis=1, keepdims=True)
    U = U / np.where(r > 0, r, 1.0)

    return U, centers, history


# =============================================================================
# 6.  PUBLIC FCM WRAPPER  (best-of-n_runs)
# =============================================================================

def run_fcm(W, C, m=2.0, n_runs=5, max_iter=500,
            tol=1e-7, random_state=42, verbose=True):
    """
    Run FCM `n_runs` times and return the solution with the lowest
    final objective J (most stable solution).

    Parameters
    ----------
    W            : np.ndarray (N, N)  Preprocessed adjacency matrix.
    C            : int                Number of communities.
    m            : float              Fuzziness exponent.
    n_runs       : int                Number of independent restarts.
    max_iter     : int
    tol          : float
    random_state : int
    verbose      : bool

    Returns
    -------
    U       : np.ndarray (N, C)   Best membership matrix found.
    history : list[float]          Objective history of the best run.
    """
    rng     = np.random.default_rng(random_state)
    best_U, best_hist, best_J = None, None, np.inf

    for run in range(n_runs):
        seed = int(rng.integers(0, 100_000))
        U, _, hist = _fcm(W, C=C, m=m, max_iter=max_iter, tol=tol,
                          random_state=seed,
                          verbose=(verbose and run == 0))
        if hist[-1] < best_J:
            best_J, best_U, best_hist = hist[-1], U, hist

    if verbose:
        print(f"[fcm] Best J = {best_J:.4f} over {n_runs} runs.\n")
    return best_U, best_hist


# =============================================================================
# 7.  COMMUNITY PARTICIPATION SIMILARITY MATRIX  S_JS
# =============================================================================

def compute_SJS(U):
    """
    Build the Community Participation Similarity Matrix S_JS.

    For each node pair (i, j):
        JS_ij  = Jensen-Shannon divergence(u_i ‖ u_j)   ∈ [0, 1]
        d_ij   = sqrt(JS_ij)      (Jensen-Shannon metric, triangle-ineq. holds)
        S_JS_ij = 1 − d_ij        ∈ [0, 1]   (1 = identical profiles)

    Membership vectors are first clipped to [ε, 1] and renormalised to
    form valid probability distributions, as required by JS divergence.

    Parameters
    ----------
    U : np.ndarray (N, C)   Membership matrix; rows should sum to 1.

    Returns
    -------
    S_JS : np.ndarray (N, N)  Similarity matrix, symmetric, diagonal = 1.
    JS   : np.ndarray (N, N)  Raw JS divergence, symmetric, diagonal = 0.
    """
    N = U.shape[0]

    # Sanitise: clip tiny negatives, renormalise rows
    U_p = np.clip(U, 1e-10, 1.0)
    U_p = U_p / U_p.sum(axis=1, keepdims=True)

    JS   = np.zeros((N, N))
    S_JS = np.eye(N)          # diagonal = 1 by definition

    for i in range(N):
        for j in range(i + 1, N):
            # jensenshannon() returns sqrt(JS divergence) in [0, 1]
            js_metric      = float(jensenshannon(U_p[i], U_p[j], base=2))
            js_div         = js_metric ** 2
            JS[i, j]       = JS[j, i] = js_div
            sim            = 1.0 - js_metric
            S_JS[i, j]     = S_JS[j, i] = sim

    return S_JS, JS


# =============================================================================
# 8.  BRIDGE NODE DETECTION
# =============================================================================

def detect_bridge_nodes(U, roi_labels=None, threshold_pct=70):
    """
    Flag bridge nodes via Shannon entropy of membership vectors.

    A bridge node has membership spread across multiple communities
    (high entropy).  A core node is concentrated in one community
    (low entropy).

    Entropy:      H_i   = −Σ_k u_ik log₂ u_ik
    Max entropy:  H_max =  log₂ C   (uniform across all communities)
    Threshold:    nodes with H_i > (threshold_pct / 100) × H_max

    Parameters
    ----------
    U             : np.ndarray (N, C)   Membership matrix.
    roi_labels    : list[int] | None    ROI labels for readable output.
    threshold_pct : float               Entropy % of H_max (default 70).

    Returns
    -------
    info : dict with keys
        'entropy'            np.ndarray (N,)   Entropy per node (bits).
        'norm_entropy'       np.ndarray (N,)   H_i / H_max  ∈ [0, 1].
        'is_bridge'          np.ndarray (N,)   Boolean mask.
        'bridge_indices'     np.ndarray        Integer indices of bridges.
        'bridge_labels'      list[int]         ROI labels of bridges.
        'threshold'          float             Entropy threshold used.
        'dominant_community' np.ndarray (N,)   argmax community per node.
    """
    N, C  = U.shape
    H_max = np.log2(C)

    U_safe  = np.clip(U, 1e-10, 1.0)
    entropy = -np.sum(U_safe * np.log2(U_safe), axis=1)   # (N,)
    norm_H  = entropy / H_max

    threshold    = (threshold_pct / 100.0) * H_max
    is_bridge    = entropy > threshold
    bridge_idx   = np.where(is_bridge)[0]
    bridge_lbls  = ([roi_labels[i] for i in bridge_idx]
                    if roi_labels is not None else [])
    dominant     = np.argmax(U, axis=1)

    print(f"[bridges] H_max={H_max:.3f} bits | "
          f"threshold={threshold:.3f} bits ({threshold_pct}%)")
    print(f"[bridges] Bridge nodes: {is_bridge.sum()} / {N} "
          f"({100*is_bridge.mean():.1f}%)")
    if bridge_lbls:
        print(f"[bridges] ROI labels: {bridge_lbls}")

    return {
        'entropy':            entropy,
        'norm_entropy':       norm_H,
        'is_bridge':          is_bridge,
        'bridge_indices':     bridge_idx,
        'bridge_labels':      bridge_lbls,
        'threshold':          threshold,
        'dominant_community': dominant,
    }


# =============================================================================
# 9.  FULL SINGLE-SUBJECT PIPELINE
# =============================================================================

def run_pipeline(fc_path_or_matrix,
                 C_min=2, C_max=10,
                 m=2.0, n_runs=5,
                 threshold_pct=10,
                 negative_strategy='zero',
                 fisher_z=True,
                 random_state=42,
                 verbose=True,
                 node_number=None
                 ):
    """
    End-to-end pipeline for one subject.

    Parameters
    ----------
    fc_path_or_matrix : str | np.ndarray
        Path to a subject CSV file, or a pre-loaded (N,N) FC matrix.
        If a matrix is passed, roi_labels will be None.
    C_min, C_max      : int     Community-number search range.
    m                 : float   FCM fuzziness exponent (default 2.0).
    n_runs            : int     FCM restarts per C (for stability).
    threshold_pct     : float   Proportional threshold (keep top X% edges).
    negative_strategy : str     'zero' | 'absolute' | 'keep'.
    fisher_z          : bool    Apply Fisher r-to-z transform (recommended).
    random_state      : int
    verbose           : bool

    Returns
    -------
    results : dict
        'U'           np.ndarray (N, C)   Membership matrix
        'S_JS'        np.ndarray (N, N)   Similarity matrix
        'JS'          np.ndarray (N, N)   JS divergence matrix
        'W'           np.ndarray (N, N)   Preprocessed adjacency matrix
        'C'           int                 Optimal number of communities
        'Q_scores'    dict[int,float]     Modularity per C
        'roi_labels'  list[int] | None    ROI labels
        'bridge_info' dict                Bridge-node analysis
        'history'     list[float]         FCM objective convergence
    """
    bar = "=" * 65
    if verbose:
        print(f"\n{bar}\nFuzzy C-Means Brain Network Pipeline\n{bar}")

    # ── Load ──────────────────────────────────────────────────────────
    if isinstance(fc_path_or_matrix, str):
        FC, roi_labels = load_fc_matrix(fc_path_or_matrix, node_number)
    else:
        FC, roi_labels = fc_path_or_matrix.copy(), None

    # ── Preprocess ────────────────────────────────────────────────────
    W, _ = preprocess_fc_matrix(
        FC,
        threshold_pct=threshold_pct,
        negative_strategy=negative_strategy,
        fisher_z=fisher_z,
    )

    # ── Optimal C ─────────────────────────────────────────────────────
    best_C, Q_scores = find_optimal_C(
        W, C_min=C_min, C_max=C_max,
        n_runs=n_runs, m_fuzzy=m,
        random_state=random_state,
    )

    # ── Final FCM ─────────────────────────────────────────────────────
    if verbose:
        print(f"[fcm] Final run: C={best_C}, m={m}, n_runs={n_runs}")
    U, history = run_fcm(
        W, C=best_C, m=m, n_runs=n_runs,
        random_state=random_state + 1,
        verbose=verbose,
    )

    # ── S_JS ──────────────────────────────────────────────────────────
    if verbose:
        print("[SJS] Computing Community Participation Similarity Matrix …")
    S_JS, JS = compute_SJS(U)

    # ── Bridge nodes ──────────────────────────────────────────────────
    bridge_info = detect_bridge_nodes(U, roi_labels=roi_labels)

    if verbose:
        N = U.shape[0]
        print(f"\n{bar}")
        print(f"Done | N={N} | C={best_C} | "
              f"bridges={bridge_info['is_bridge'].sum()}")
        print(f"S_JS ∈ [{S_JS[S_JS<1].min():.4f}, "
              f"{S_JS[S_JS<1].max():.4f}]")
        print(bar)

    return {
        'U':           U,
        'S_JS':        S_JS,
        'JS':          JS,
        'W':           W,
        'C':           best_C,
        'Q_scores':    Q_scores,
        'roi_labels':  roi_labels,
        'bridge_info': bridge_info,
        'history':     history,
    }


# =============================================================================
# 10.  GROUP-LEVEL STATISTICAL ANALYSIS
# =============================================================================

def group_analysis(SJS_list, group_labels,
                   roi_labels=None, alpha=0.05):
    """
    One-way ANOVA on each S_JS element across diagnostic groups,
    followed by Benjamini-Hochberg FDR correction.

    Intended use: compare HC vs EMCI vs LMCI vs AD.

    Parameters
    ----------
    SJS_list     : list[np.ndarray (N,N)]   One matrix per subject.
    group_labels : list[str | int]           Group per subject.
    roi_labels   : list[int] | None          ROI labels for output.
    alpha        : float                     FDR significance level.

    Returns
    -------
    stats : dict
        'F_matrix'  np.ndarray (N,N)   F-statistic per element.
        'p_raw'     np.ndarray (N,N)   Uncorrected p-values.
        'p_fdr'     np.ndarray (N,N)   BH-corrected p-values.
        'sig_mask'  np.ndarray (N,N)   Boolean (significant after FDR).
        'sig_pairs' list[tuple]        (roi_i, roi_j, F, p_fdr) tuples.
    """
    from scipy.stats import f_oneway
    try:
        from statsmodels.stats.multitest import multipletests
        _sm = True
    except ImportError:
        _sm = False
        print("[anova] statsmodels not found — using Bonferroni correction.\n"
              "        Install with: pip install statsmodels")

    groups = sorted(set(group_labels))
    by_grp = {g: [] for g in groups}
    for mat, grp in zip(SJS_list, group_labels):
        by_grp[grp].append(mat)
    for g in groups:
        by_grp[g] = np.stack(by_grp[g])    # (n_subj, N, N)

    N        = SJS_list[0].shape[0]
    F_mat    = np.zeros((N, N))
    p_mat    = np.ones((N, N))

    print(f"[anova] Running one-way ANOVA across {groups} …")
    for i in range(N):
        for j in range(i + 1, N):
            samples = [by_grp[g][:, i, j] for g in groups]
            if all(len(s) > 1 for s in samples):
                F, p = f_oneway(*samples)
                F_mat[i, j] = F_mat[j, i] = F
                p_mat[i, j] = p_mat[j, i] = p

    # FDR on upper-triangle p-values
    uidx  = np.triu_indices(N, k=1)
    p_up  = p_mat[uidx]

    if _sm:
        _, p_fdr_up, _, _ = multipletests(p_up, alpha=alpha, method='fdr_bh')
    else:
        p_fdr_up = np.minimum(p_up * len(p_up), 1.0)   # Bonferroni

    p_fdr               = np.ones((N, N))
    p_fdr[uidx]         = p_fdr_up
    p_fdr[uidx[1],uidx[0]] = p_fdr_up      # symmetrise

    sig_mask  = p_fdr < alpha
    sig_pairs = []
    for i, j in zip(*np.where(sig_mask & np.triu(np.ones((N,N),bool), k=1))):
        li = roi_labels[i] if roi_labels else i
        lj = roi_labels[j] if roi_labels else j
        sig_pairs.append((li, lj, float(F_mat[i,j]), float(p_fdr[i,j])))

    print(f"[anova] Significant pairs (FDR<{alpha}): "
          f"{len(sig_pairs)} / {len(uidx[0])}")
    return {
        'F_matrix':  F_mat,
        'p_raw':     p_mat,
        'p_fdr':     p_fdr,
        'sig_mask':  sig_mask,
        'sig_pairs': sig_pairs,
    }


# =============================================================================
# 11.  VISUALISATION
# =============================================================================

def visualize_results(results, subject_id=None, save_path=None):
    """
    Six-panel diagnostic figure for one subject's pipeline output.

    Panels
    ------
    1. Preprocessed FC matrix W
    2. Modularity Q vs. C  (optimal C marked)
    3. Membership matrix U  (N × C heat-map)
    4. S_JS similarity matrix
    5. Node entropy bar chart  (bridge nodes highlighted)
    6. FCM objective convergence curve

    Parameters
    ----------
    results    : dict   Output of run_pipeline().
    subject_id : str    Label for figure title (optional).
    save_path  : str    Save to file if given, else plt.show().
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    U        = results['U']
    S_JS     = results['S_JS']
    W        = results['W']
    C        = results['C']
    Q_scores = results['Q_scores']
    history  = results['history']
    bridge   = results['bridge_info']
    N        = U.shape[0]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    def sax(ax, title):
        ax.set_facecolor('#161b22')
        ax.set_title(title, color='#e6edf3', fontsize=11,
                     fontweight='bold', pad=10)
        for sp in ax.spines.values():
            sp.set_edgecolor('#30363d')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        return ax

    def cbar(im, ax):
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color='#8b949e', labelcolor='#8b949e')
        cb.outline.set_edgecolor('#30363d')

    # 1 — W
    ax = sax(fig.add_subplot(gs[0, 0]), "Preprocessed FC Matrix  W")
    im = ax.imshow(W, cmap='RdYlBu_r', aspect='auto')
    cbar(im, ax)
    ax.set_xlabel("Region index")
    ax.set_ylabel("Region index")

    # 2 — Q vs C
    ax = sax(fig.add_subplot(gs[0, 1]),
             "Modularity Q vs Number of Communities C")
    Cs = sorted(Q_scores); Qs = [Q_scores[c] for c in Cs]
    ax.plot(Cs, Qs, 'o-', color='#58a6ff', lw=2, ms=9,
            mfc='#0d1117', mew=2)
    ax.axvline(C, color='#f85149', ls='--', lw=1.5,
               label=f'Optimal C = {C}')
    ax.fill_between(Cs, Qs, min(Qs), alpha=0.10, color='#58a6ff')
    ax.set_xlabel("C"); ax.set_ylabel("Q")
    ax.set_xticks(Cs)
    ax.legend(facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#e6edf3', fontsize=9)

    # 3 — U
    ax = sax(fig.add_subplot(gs[0, 2]),
             f"Membership Matrix  U  (N={N}, C={C})")
    im = ax.imshow(U, cmap='magma', aspect='auto', vmin=0, vmax=1)
    cbar(im, ax)
    ax.set_xlabel("Community")
    ax.set_ylabel("Brain region")
    ax.set_xticks(range(C))
    ax.set_xticklabels([f"C{k+1}" for k in range(C)])

    # 4 — S_JS
    ax = sax(fig.add_subplot(gs[1, 0]),
             "Community Participation Similarity  S_JS")
    im = ax.imshow(S_JS, cmap='plasma', aspect='auto', vmin=0, vmax=1)
    cbar(im, ax)
    ax.set_xlabel("Region index"); ax.set_ylabel("Region index")

    # 5 — Entropy / bridge nodes
    ax  = sax(fig.add_subplot(gs[1, 1]),
              "Node Entropy  (red = bridge node)")
    ent = bridge['entropy']
    thr = bridge['threshold']
    col = ['#f85149' if b else '#58a6ff' for b in bridge['is_bridge']]
    ax.bar(range(N), ent, color=col, width=0.8, alpha=0.85)
    ax.axhline(thr, color='#f85149', ls='--', lw=1.5,
               label=f"threshold  ({bridge['is_bridge'].sum()} bridges)")
    ax.set_xlabel("Region index")
    ax.set_ylabel("Shannon entropy (bits)")
    ax.legend(facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#e6edf3', fontsize=9)

    # 6 — Convergence
    ax = sax(fig.add_subplot(gs[1, 2]), "FCM Objective Convergence")
    ax.plot(history, color='#3fb950', lw=2)
    ax.fill_between(range(len(history)), history,
                    min(history), alpha=0.15, color='#3fb950')
    ax.set_xlabel("Iteration"); ax.set_ylabel("Objective J")

    title = "Fuzzy C-Means Brain Network Analysis"
    if subject_id:
        title += f"  |  {subject_id}"
    title += f"  |  C={C}  |  N={N} regions"
    fig.suptitle(title, color='#e6edf3', fontsize=13,
                 fontweight='bold', y=0.99)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"[viz] Saved → {save_path}")
        plt.close()
    else:
        plt.show()
