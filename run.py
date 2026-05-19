import numpy as np
from fuzzy_cmeans_brain import preprocess_fc_matrix, run_pipeline

FC = np.load('your_fc_matrix.npy')   # shape (116, 116) for AAL atlas
W  = preprocess_fc_matrix(FC, threshold_pct=10)
results = run_pipeline(W, C_min=2, C_max=10)

U    = results['U']      # membership matrix (116 × C)
S_JS = results['S_JS']   # similarity matrix (116 × 116)