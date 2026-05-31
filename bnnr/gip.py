import numpy as np

def getGIPSim(Adm_interaction, gamma0_d=None, gamma0_m=None, AvoidIsolatedNodes=True, RemoveNonoverlapPairs=True):
    """
    Compute Gaussian Interaction Profile (GIP) kernel similarity matrices.

    Ref: Gaussian interaction profile kernels for predicting drug-target interaction

    :param Adm_interaction: interaction matrix (rows: disease/drug, cols: miRNA/target)
    :param gamma0_d: row-dimension kernel bandwidth (scalar/None)
    :param gamma0_m: column-dimension kernel bandwidth (scalar/None)
    :param AvoidIsolatedNodes: filter isolated nodes (default True)
    :param RemoveNonoverlapPairs: zero out non-overlapping pairs (default True)
    :return: kd, km -- row/col GIP similarity matrices (None if corresponding gamma is None)
    """
    Adm_interaction = np.array(Adm_interaction, dtype=np.float64)

    if gamma0_d is None and gamma0_m is None:
        raise ValueError('both gamma0_d and gamma0_m are empty. No output.')

    nd_all, nm_all = Adm_interaction.shape

    if AvoidIsolatedNodes:
        nodes_d = np.sum(Adm_interaction, axis=1) != 0
        nodes_m = np.sum(Adm_interaction, axis=0) != 0
        Adm = Adm_interaction[nodes_d, :][:, nodes_m].copy()
    else:
        Adm = Adm_interaction.copy()

    nd, nm = Adm.shape
    SumOfSquares = np.sum(np.square(Adm))

    # Row-entity similarity (kd)
    if gamma0_d is not None:
        gamma0_d = gamma0_d / (SumOfSquares / nd)
        D = Adm @ Adm.T
        dd = np.diag(D)
        kd2 = np.exp(-gamma0_d * (dd[:, np.newaxis] + dd[np.newaxis, :] - 2 * D))
        if RemoveNonoverlapPairs:
            kd2[~D.astype(bool)] = 0.0

        if AvoidIsolatedNodes:
            kd = np.zeros((nd_all, nd_all), dtype=np.float64)
            kd[nodes_d, :][:, nodes_d] = kd2
        else:
            kd = kd2
    else:
        kd = None

    # Column-entity similarity (km)
    if gamma0_m is not None:
        gamma0_m = gamma0_m / (SumOfSquares / nm)
        E = Adm.T @ Adm
        mm = np.diag(E)
        km2 = np.exp(-gamma0_m * (mm[:, np.newaxis] + mm[np.newaxis, :] - 2 * E))
        if RemoveNonoverlapPairs:
            km2[~E.astype(bool)] = 0.0

        if AvoidIsolatedNodes:
            km = np.zeros((nm_all, nm_all), dtype=np.float64)
            km[nodes_m, :][:, nodes_m] = km2
        else:
            km = km2
    else:
        km = None

    return kd, km
