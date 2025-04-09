import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot if ss_tot != 0 else 0

def d_spls_metric(model, X, y):
    """
    Computes performance metrics for a dual-SPLS model.
    
    Parameters
    ----------
    model : dict
         Model dictionary that includes 'Bhat' and 'intercept'.
    X : np.ndarray
         The predictor matrix.
    y : np.ndarray
         The response vector.
    
    Returns
    -------
    dict
         Contains MSE and R2 for each component.
    """
    ncomp = model['Bhat'].shape[1]
    mse_values = []
    r2_values = []
    for comp in range(ncomp):
        pred = X @ model['Bhat'][:, comp] + model['intercept'][comp]
        mse_values.append(mse(y, pred))
        r2_values.append(r2_score(y, pred))
    return {'MSE': mse_values, 'R2': r2_values}
