import numpy as np

def _kfold_indices(n_samples, nfold):
    """Returns a vector of fold assignments for n_samples samples."""
    if n_samples == 0:
        return np.array([], dtype=int)
    base_size = n_samples // nfold
    reminder = n_samples % nfold
    fold_sizes = [base_size + 1 if i < reminder else base_size for i in range(nfold)]
    fold_assign = []
    for fold in range(nfold):
        fold_assign.extend([fold] * fold_sizes[fold])
    fold_assign = np.array(fold_assign, dtype=int)
    np.random.shuffle(fold_assign)
    return fold_assign

def _find_linear_idx(mat, val):
    mat = np.array(mat)
    row_idx, col_val = np.where(mat == val)
    linear_idx = np.ravel_multi_index((row_idx, col_val), mat.shape, order='F')
    return linear_idx

def getKfoldCrossValidMatIndSet(Wdr, nfold, CVtype, NegType=None, seed=None):
    """
    Generate K-fold cross-validation index sets.

    Supports three CV strategies:
      - CVa: random split by association pairs
      - CVr: row-wise split (hide all associations for some drugs)
      - CVc: column-wise split (hide all associations for some diseases)

    Returns dict with keys:
      MatIndSet_pos_test, MatIndSet_neg_test,
      MatIndSet_pos_train, MatIndSet_neg_train
    """
    if seed is not None:
        np.random.seed(seed)

    IndSet_pos_test = [[] for _ in range(nfold)]
    IndSet_neg_test = [[] for _ in range(nfold)]
    IndSet_pos_train = [[] for _ in range(nfold)]
    IndSet_neg_train = [[] for _ in range(nfold)]

    Wdr = np.array(Wdr, dtype=float)
    Wdr_mark = np.where(Wdr != 0, 1, -1)

    if CVtype == 'CVa':
        ind_pos = _find_linear_idx(Wdr_mark, 1)
        ind_neg = _find_linear_idx(Wdr_mark, -1)
        ipos = _kfold_indices(len(ind_pos), nfold) if len(ind_pos) > 0 else np.array([])
        ineg = _kfold_indices(len(ind_neg), nfold) if len(ind_neg) > 0 else np.array([])

        for fold in range(nfold):
            if len(ind_pos) > 0:
                IndSet_pos_test[fold] = ind_pos[ipos == fold]
                IndSet_pos_train[fold] = ind_pos[ipos != fold]

            if NegType is not None and NegType.casefold() == 'Unlabel'.casefold():
                IndSet_neg_test[fold] = ind_neg
                IndSet_neg_train[fold] = np.union1d(IndSet_neg_test[fold], IndSet_pos_test[fold])
            else:
                if len(ind_neg) > 0:
                    IndSet_neg_test[fold] = ind_neg[ineg == fold]
                    IndSet_neg_train[fold] = ind_neg[ineg != fold]

    elif CVtype == 'CVr':
        k = np.any(Wdr_mark == 1, axis=1)
        rInd = np.where(k)[0]
        ii = _kfold_indices(len(rInd), nfold) if len(rInd) > 0 else np.array([])

        for fold in range(nfold):
            mat_test = np.zeros_like(Wdr_mark)
            if len(rInd) > 0:
                row_idx = rInd[ii == fold]
                mat_test[row_idx, :] = Wdr_mark[row_idx, :]
            ind_pos_test = _find_linear_idx(mat_test, 1)
            ind_neg_test = _find_linear_idx(mat_test, -1)
            if ind_neg_test.size == 0:
                ind_neg_test = _find_linear_idx(Wdr_mark, -1)
            IndSet_pos_test[fold] = ind_pos_test
            IndSet_neg_test[fold] = ind_neg_test

            mat_train = Wdr_mark.copy()
            if len(rInd) > 0:
                row_idx = rInd[ii == fold]
                mat_train[row_idx, :] = 0
            ind_pos_train = _find_linear_idx(mat_train, 1)
            ind_neg_train = _find_linear_idx(mat_train, -1)
            if ind_neg_train.size == 0:
                ind_neg_train = np.union1d(IndSet_neg_test[fold], IndSet_pos_test[fold])
            IndSet_pos_train[fold] = ind_pos_train
            IndSet_neg_train[fold] = ind_neg_train

    elif CVtype == 'CVc':
        k = np.any(Wdr_mark == 1, axis=0)
        cInd = np.where(k)[0]
        ii = _kfold_indices(len(cInd), nfold) if len(cInd) > 0 else np.array([])

        for fold in range(nfold):
            mat_test = np.zeros_like(Wdr_mark)
            if len(cInd) > 0:
                col_idx = cInd[ii == fold]
                mat_test[:, col_idx] = Wdr_mark[:, col_idx]
            ind_pos_test = _find_linear_idx(mat_test, 1)
            ind_neg_test = _find_linear_idx(mat_test, -1)
            if ind_neg_test.size == 0:
                ind_neg_test = _find_linear_idx(Wdr_mark, -1)
            IndSet_pos_test[fold] = ind_pos_test
            IndSet_neg_test[fold] = ind_neg_test

            mat_train = Wdr_mark.copy()
            if len(cInd) > 0:
                col_idx = cInd[ii == fold]
                mat_train[:, col_idx] = 0
            ind_pos_train = _find_linear_idx(mat_train, 1)
            ind_neg_train = _find_linear_idx(mat_train, -1)
            if ind_neg_train.size == 0:
                ind_neg_train = np.union1d(IndSet_neg_test[fold], IndSet_pos_test[fold])
            IndSet_pos_train[fold] = ind_pos_train
            IndSet_neg_train[fold] = ind_neg_train
    else:
        raise ValueError(f"There is no cross-validation definition of : {CVtype}")

    CVdata = {
        'MatIndSet_pos_test': IndSet_pos_test,
        'MatIndSet_neg_test': IndSet_neg_test,
        'MatIndSet_pos_train': IndSet_pos_train,
        'MatIndSet_neg_train': IndSet_neg_train
    }
    return CVdata
