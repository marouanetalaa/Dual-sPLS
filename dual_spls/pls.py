import numpy as np
import dual_spls.norm as norm

def d_spls_pls(X, y, ncp, verbose=True):
    """
    Univariate Partial Least Squares (PLS1) regression using Wold's NIPALS algorithm.

    Parameters:
        X (np.ndarray): Predictor matrix of shape (n, p).
        y (np.ndarray): Response vector of shape (n,) or (n, 1).
        ncp (int): Number of components.
        verbose (bool): Whether to print iteration details.

    Returns:
        dict: Resulting model with keys:
            - Xmean, scores, loadings, Bhat, intercept, fitted_values, residuals, type
    """
    
    y = y.flatten()  # Ensure y is a 1D array
    n, p = X.shape

    # Centering
    Xm = np.mean(X, axis=0)
    Xc = X - Xm

    ym = np.mean(y)
    yc = y - ym

    # Initialization
    WW = np.zeros((p, ncp))
    TT = np.zeros((n, ncp))
    Bhat = np.zeros((p, ncp))
    YY = np.zeros((n, ncp))
    RES = np.zeros((n, ncp))
    intercept = np.zeros(ncp)

    Xdef = Xc.copy()

    for ic in range(ncp):
        Z = Xdef.T @ yc
        Z2 = norm.norm2(Z)
        mu = Z2

        w = Z / Z2
        t = Xdef @ w
        t = t / norm.norm2(t)

        WW[:, ic] = w
        TT[:, ic] = t

        Xdef = Xdef - np.outer(t, t @ Xdef)

        R = TT[:, :ic+1].T @ Xc @ WW[:, :ic+1]
        R[np.tril_indices(ic+1, -1)] = 0  # Numerical stability

        L = np.linalg.solve(R.T, np.eye(ic+1)).T  # backsolve
        Bhat[:, ic] = WW[:, :ic+1] @ (L @ (TT[:, :ic+1].T @ yc))

        intercept[ic] = ym - Xm @ Bhat[:, ic]

        YY[:, ic] = X @ Bhat[:, ic] + intercept[ic]
        RES[:, ic] = y - YY[:, ic]

        if verbose:
            print(f"PLS ic={ic+1} mu={mu:.4f}")

    return {
        'Xmean': Xm,
        'scores': TT,
        'loadings': WW,
        'Bhat': Bhat,
        'intercept': intercept,
        'fitted_values': YY,
        'residuals': RES,
        'type': 'pls'
    }
