import numpy as np
from sklearn.decomposition import PCA
from dual_spls.split import d_spls_split
from dual_spls.listecal import d_spls_listecal
from dual_spls.type import d_spls_type

def d_spls_calval(X, pcal=None, Datatype=None, y=None, ncells=10, Listecal=None,
                  center=True, method="euclidean", pc=0.9):
    """
    Splits data into calibration and validation sets using a variation of the Kennard-Stone algorithm.
    
    Parameters
    ----------
    X : np.ndarray
        Numeric matrix of predictor values (n_samples x n_features).
    pcal : float, optional
        Percentage (0–100) of calibration samples to be selected (if Listecal is not provided).
    Datatype : np.ndarray or list, optional
        Group index for each observation. If None, d_spls_type is called on y.
    y : np.ndarray, optional
        Response vector. Required if Datatype is None.
    ncells : int, default=10
        Number of groups to divide the observations.
    Listecal : list or np.ndarray, optional
        Number of samples to select from each group. If None, computed from pcal.
    center : bool, default=True
        If True, center each column of X.
    method : str, default="euclidean"
        Method for distance computation. Options are "euclidean" (default),
        "pca-euclidean" and "svd-euclidean".
    pc : float, default=0.9
        If pc < 1, interpreted as the proportion of variance to retain; otherwise the number of components.

    Returns
    -------
    dict
        Dictionary with keys:
            - 'indcal': indices of calibration samples.
            - 'indval': indices of validation samples.
    """
    n = X.shape[0]
    if center:
        Xm = np.mean(X, axis=0)
        X = X - Xm  # center X columnwise

    if Datatype is None and y is None:
        raise ValueError("If Datatype is None, y must not be None.")
    # If Datatype not provided, compute it using d_spls_type (see type.py)
    if Datatype is None:
        Datatype = d_spls_type(y, ncells)
        
    if Listecal is None and pcal is None:
        raise ValueError("Either Listecal or pcal must be provided.")
    if Listecal is None:
        Listecal = d_spls_listecal(Datatype, pcal)
    if np.max(Datatype) != len(Listecal):
        raise ValueError("Length of Listecal does not match with values of Datatype.")
        
    # Determine number of components for transformation if needed.
    if pc < 0:
        pc = 0.9
        print('Warning: pc cannot be negative; pc is set to 0.9.')
    
    if method in ["pca-euclidean", "svd-euclidean"]:
        pca = PCA()
        pca.fit(X)
        # If pc is less than 1, interpret it as variance proportion:
        if pc < 1:
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.searchsorted(cumvar, pc) + 1
        else:
            n_components = int(pc)
        if method == "pca-euclidean":
            X = pca.transform(X)[:, :n_components]
        elif method == "svd-euclidean":
            # Using np.linalg.svd: U, S, Vt = svd(X, full_matrices=False)
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X = U[:, :n_components]
    
    # Call the split function to determine calibration indices.
    indcal = d_spls_split(X, Xtype=np.array(Datatype), Listecal=Listecal)
    all_idx = np.arange(n)
    indval = np.setdiff1d(all_idx, np.array(indcal))
    
    return {'indcal': indcal, 'indval': indval}
