# BADGE — Bayesian Adaptive Drug-disease Graph Enhancement
# Integrates: BNNR, RA-BNNR, GBNNR, GBNNR-v3, GF-BNNR, BADGE

from .core import BNNR, BNNR_adaptive, BNNR_graph_aware, infer_ra_params
from .graph import (BNNR_graph, BNNR_graph_enhanced_v3,
                     build_knn_graph, normalized_laplacian,
                     normalized_laplacian_sparse)
from .filter import GF_BNNR
from .badge import BADGE
from .gip import getGIPSim
from .cv import getKfoldCrossValidMatIndSet
from .metrics import getPerfMetricROCcompute, compute_topk_metrics
from .svt import svt, svt_with_rank
from .helpers import (ensure_dir, load_dataset, mask_test_entries,
                       build_augmented_matrix, extract_recovery_block,
                       evaluate_fold)
