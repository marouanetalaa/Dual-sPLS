import numpy as np
from dual_spls.lasso import d_spls_lasso
from dual_spls.pls import d_spls_pls
from dual_spls.LS import d_spls_LS
from dual_spls.ridge import d_spls_ridge
from dual_spls.GLA import d_spls_GLA
from dual_spls.GLB import d_spls_GLB
from dual_spls.GLC import d_spls_GLC

def d_spls_errorcv(cv_indices, X, Y, ncomp, dspls="lasso", ppnu=None, nu2=None, indG=None, gamma=None):
    """
    Computes cross-validation error (MSE) for a single calibration split.
    
    Parameters
    ----------
    cv_indices : array-like
        Indices of the calibration set.
    X : np.ndarray
        Data matrix.
    Y : np.ndarray
        Response vector.
    ncomp : int
        Number of latent components to fit.
    dspls : str, default="lasso"
        The type of dual-SPLS regression to use.
    ppnu : float or list
        Parameter controlling shrinkage.
    nu2 : float
        Parameter used for ridge norm (if applicable).
    indG : array-like, optional
        Group indices (for group lasso norms).
    gamma : list or np.ndarray, optional
        Gamma weights (for norm C).
    
    Returns
    -------
    np.ndarray
        An array of mean squared errors for components 1 to ncomp.
    """
    n = X.shape[0]
    all_idx = np.arange(n)
    val_idx = np.setdiff1d(all_idx, np.array(cv_indices))
    X_cal = X[cv_indices, :]
    Y_cal = Y[cv_indices]
    X_val = X[val_idx, :]
    Y_val = Y[val_idx]
    
    # Dispatch to the appropriate dual-SPLS function.
    if dspls.lower() == "lasso":
        model = d_spls_lasso(X_cal, Y_cal, ncp=ncomp, ppnu=ppnu, verbose=False)
    elif dspls.lower() == "pls":
        model = d_spls_pls(X_cal, Y_cal, ncp=ncomp, verbose=False)
    elif dspls.upper() == "LS":
        model = d_spls_LS(X_cal, Y_cal, ncp=ncomp, ppnu=ppnu, verbose=False)
    elif dspls.lower() == "ridge":
        model = d_spls_ridge(X_cal, Y_cal, ncp=ncomp, ppnu=ppnu, verbose=False)
    elif dspls.upper() == "GLA":
        model = d_spls_GLA(X_cal, Y_cal, ncp=ncomp, ppnu=ppnu, indG=indG, verbose=False)
    elif dspls.upper() == "GLB":
        model = d_spls_GLB(X_cal, Y_cal, ncp=ncomp, ppnu=ppnu, indG=indG, verbose=False)
    elif dspls.upper() == "GLC":
        model = d_spls_GLC(X_cal, Y_cal, ncp=ncomp, ppnu=ppnu, indG=indG, gamma=gamma, verbose=False)
    else:
        raise ValueError("Unknown dspls type.")
    
    errors = np.zeros(ncomp)
    # For each component from 1 to ncomp, compute prediction error (MSE)
    for comp in range(ncomp):
        pred = X_val @ model['Bhat'][:, comp] + model['intercept'][comp]
        errors[comp] = np.mean((Y_val - pred)**2)
    return errors

def d_spls_cv(X, Y, ncomp, dspls="lasso", ppnu=None, nu2=None, nrepcv=30, pctcv=70, indG=None, gamma=None):
    """
    Determines the optimal number of latent components via cross validation.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    Y : np.ndarray
        Response vector.
    ncomp : int or array-like
        Either the maximum number of components to try (if int) or a list of component numbers.
    dspls : str, default="lasso"
        The norm type to use.
    ppnu : float or list
        Parameter for variable shrinkage.
    nu2 : float
        Constraint parameter for the ridge norm.
    nrepcv : int, default=30
        Number of cross-validation iterations.
    pctcv : float, default=70
        Percentage of observations to use for calibration in each iteration.
    indG : array-like, optional
        Group indices (for group lasso norms).
    gamma : list or np.ndarray, optional
        Gamma weights (for norm C).
        
    Returns
    -------
    int
        The optimal number of latent components (in 1-indexing, as in the original R code).
    """
    n = X.shape[0]
    ncal = int(np.floor(n * pctcv / 100))
    # If ncomp is a single int, try all from 1 to ncomp.
    if np.isscalar(ncomp):
        if ncomp == 1:
            return 1
        comp_range = np.arange(1, ncomp + 1)
    else:
        comp_range = np.array(ncomp)
        ncomp = len(comp_range)
    
    # Generate calibration indices for each replication (each column is one split)
    cv_cal = [np.random.choice(n, ncal, replace=False) for _ in range(nrepcv)]
    
    # Collect errors for each replication (rows: component number, columns: replication)
    errorcv = np.zeros((len(comp_range), nrepcv))
    for i, cv_idx in enumerate(cv_cal):
        err = d_spls_errorcv(cv_idx, X, Y, len(comp_range), dspls=dspls, ppnu=ppnu, nu2=nu2,
                             indG=indG, gamma=gamma)
        errorcv[:, i] = err
    # Compute mean error for each component (across replications)
    mean_error = np.mean(errorcv, axis=1)
    # Choose the component number (from comp_range) with the smallest mean error
    opt_comp = comp_range[np.argmin(mean_error)]
    return int(opt_comp)
