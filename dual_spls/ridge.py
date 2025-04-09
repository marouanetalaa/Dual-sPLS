import numpy as np
from scipy.linalg import lstsq

def d_spls_ridge(X, y, ncp, ppnu, verbose=True):
    """
    Dual-SPLS Ridge regression.
    
    Parameters
    ----------
    X : np.ndarray
        Predictor matrix.
    y : np.ndarray
        Response vector.
    ncp : int
        Number of components.
    ppnu : float
        Parameter controlling shrinkage.
    verbose : bool, optional
        If True, prints iteration details.
    
    Returns
    -------
    dict
        Dictionary with model components.
    """
    n, p = X.shape
    Xm = np.mean(X, axis=0)
    Xc = X - Xm
    ym = np.mean(y)
    yc = y - ym
    
    WW = np.zeros((p, ncp))
    TT = np.zeros((n, ncp))
    Bhat = np.zeros((p, ncp))
    intercept = np.zeros(ncp)
    zerovar = np.zeros(ncp, dtype=int)
    
    # This example uses a simple ridge-like step in dual-SPLS
    X_def = Xc.copy()
    for ic in range(ncp):
        Z = X_def.T @ yc
        # Ridge shrinkage; note: nu and ppnu here are used analogously
        nu = ppnu  # placeholder for ridge parameter, you might adjust this logic
        Znu = np.sign(Z) * np.maximum(np.abs(Z) - nu, 0)
        # Compute weight vector (simple scaling)
        w = Znu / (np.linalg.norm(Znu)+1e-8)
        WW[:, ic] = w
        t = X_def @ w
        t_norm = np.linalg.norm(t)
        if t_norm != 0:
            t = t / t_norm
        TT[:, ic] = t
        # Deflation
        X_def = X_def - np.outer(t, t.T @ X_def)
        # Regression coefficients computation:
        R = TT[:, :ic+1].T @ Xc @ WW[:, :ic+1]
        R[np.tril_indices(R.shape[0])] = 0
        L, _, _, _ = lstsq(R, np.eye(ic+1))
        Bhat[:, ic] = WW[:, :ic+1] @ (L @ (TT[:, :ic+1].T @ yc))
        intercept[ic] = ym - Xm @ Bhat[:, ic]
        zerovar[ic] = np.sum(Bhat[:, ic] == 0)
        if verbose:
            print(f"Ridge: Component {ic+1} computed.")
    
    return {'Xmean': Xm, 'scores': TT, 'loadings': WW, 'Bhat': Bhat,
            'intercept': intercept, 'fitted_values': X @ Bhat + intercept,
            'zerovar': zerovar,
            'type': 'ridge'}
