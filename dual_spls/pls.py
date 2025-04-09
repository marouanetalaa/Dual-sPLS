import numpy as np
from scipy.linalg import lstsq

def d_spls_pls(X, y, ncp, verbose=False):
    """
    Simple Partial Least Squares (PLS) implementation.
    (This version is a placeholder; in practice one may want to use packages like sklearn.cross_decomposition.PLSRegression.)
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    y : np.ndarray
        Response vector.
    ncp : int
        Number of components.
    verbose : bool, optional
        Print progress if True.
    
    Returns
    -------
    dict
        Dictionary with keys 'scores', 'loadings', 'Bhat', 'intercept', and others.
    """
    n, p = X.shape
    # Centering
    Xm = np.mean(X, axis=0)
    Xc = X - Xm
    ym = np.mean(y)
    yc = y - ym
    
    T = np.zeros((n, ncp))
    W = np.zeros((p, ncp))
    Bhat = np.zeros((p, ncp))
    intercept = np.zeros(ncp)
    
    X_def = Xc.copy()
    
    for comp in range(ncp):
        # Simple approach: choose w proportional to X_def^T yc
        w = X_def.T @ yc
        w = w / np.linalg.norm(w) if np.linalg.norm(w)>0 else w
        t = X_def @ w
        t = t / np.linalg.norm(t) if np.linalg.norm(t)>0 else t
        T[:, comp] = t
        W[:, comp] = w
        
        # Deflation
        X_def = X_def - np.outer(t, t.T @ X_def)
        # Regression coefficients (using lstsq)
        R = T[:, :comp+1].T @ Xc @ W[:, :comp+1]
        R[np.tril_indices(R.shape[0])] = 0  # enforcing upper triangular
        L, _, _, _ = lstsq(R, np.eye(comp+1))
        Bhat[:, comp] = W[:, :comp+1] @ (L @ (T[:, :comp+1].T @ yc))
        intercept[comp] = ym - Xm @ Bhat[:, comp]
        
        if verbose:
            print(f"PLS: component {comp+1} computed.")
    
    return {'Xmean': Xm, 'scores': T, 'loadings': W, 'Bhat': Bhat, 'intercept': intercept,
            'fitted_values': X @ Bhat + intercept, 'type': 'pls'}
