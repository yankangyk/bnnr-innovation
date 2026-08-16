# GMC — Graph Multi-view Completion (final method).
#
# Optimal model under CVa (random-entry masking = matrix completion):
#   cold-start-restricted WKNN fill → multi-view low-rank completion
#   (block nuclear-norm + tensor RPCA) → rank-normalized fusion →
#   KNN-sparsified bilateral graph Laplacian filter.
# GMC-E (gmc/ensemble.py) is the score-level combination of GMC with the
# complementary low-rank baselines.
#
# Core API: gmc_predict (gmc/model.py), the graph/factorization building
# blocks in gmc/{filter,wknn,factorization}.py, and the run/ensemble drivers
# in scripts/{run_gmc,run_ensemble}.py.

from .filter import normalised_laplacian, graph_filter, sparsify_graph
from .wknn import knn_network, wknn
from .model import gmc_predict, coldstart_fill, rnorm01
from .cv import getKfoldCrossValidMatIndSet
from .metrics import getPerfMetricROCcompute, compute_topk_metrics
from .helpers import (load_dataset, load_sim_lists, mask_test_entries,
                      evaluate_fold, DATA_DIR, FOLD_DIR, OUT_DIR, RESULT_DIR)
from .factorization import (bounded_nn_completion, bounded_nn_completion_graph,
                            fitrpca, graph_reg_nmf, deep_semi_nmf)
from .ensemble import (find_pred_dir, load_fold_preds, rank_avg,
                       build_ensemble, materialize_ensemble)
