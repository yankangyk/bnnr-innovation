# GRMC — Graph-Regularized Matrix Completion
# Single-pass matrix completion with graph Laplacian regularization,
# using raw structural similarity graphs (no GIP fusion).

from .core import BNNR, BNNR_adaptive, BNNR_graph_aware, infer_ra_params
from .graph import (BNNR_graph, GBNNR,
                     build_knn_graph, normalized_laplacian,
                     normalized_laplacian_sparse)
from .filter import GF_BNNR, normalised_laplacian, graph_filter, sparsify_graph
from .badge import GRMC, BADGE
from .cv import getKfoldCrossValidMatIndSet
from .metrics import getPerfMetricROCcompute, compute_topk_metrics
from .svt import svt, svt_with_rank
from .helpers import (ensure_dir, load_dataset, mask_test_entries,
                       build_augmented_matrix, extract_recovery_block,
                       evaluate_fold)
