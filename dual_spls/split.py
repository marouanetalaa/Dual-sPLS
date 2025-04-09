import numpy as np
from sklearn.metrics import pairwise_distances

def d_spls_split(X, Xtype, Listecal):
    """
    Splits the observations into a calibration set based on the provided group information.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    Xtype : array-like
        Group type for each observation.
    Listecal : list
        For each group (in the sorted order), number of samples to select.
    
    Returns
    -------
    list
        Indices (row numbers) chosen for calibration.
    """
    X = np.array(X)
    Xtype = np.array(Xtype)
    indcal = []
    # For each group, select the observations that are most spread (Kennard-Stone like)
    for group, ncal in zip(np.unique(Xtype), Listecal):
        group_idx = np.where(Xtype == group)[0]
        if len(group_idx) <= ncal:
            indcal.extend(group_idx.tolist())
        else:
            # Compute distance matrix within the group
            dists = pairwise_distances(X[group_idx, :])
            # Start by selecting the observation furthest from the mean
            group_mean = np.mean(X[group_idx, :], axis=0)
            d_from_mean = np.linalg.norm(X[group_idx, :] - group_mean, axis=1)
            first = group_idx[np.argmax(d_from_mean)]
            sel = [first]
            remain = list(set(group_idx) - set(sel))
            while len(sel) < ncal:
                dmin = np.array([min(np.linalg.norm(X[r] - X[sel], axis=1)) for r in remain])
                next_idx = remain[np.argmax(dmin)]
                sel.append(next_idx)
                remain = list(set(remain) - {next_idx})
            indcal.extend(sel)
    return sorted(indcal)
