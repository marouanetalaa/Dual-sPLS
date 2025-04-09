import numpy as np

def d_spls_predict(model, X_new):
    """
    Makes predictions using a fitted dual-SPLS model.
    
    Parameters
    ----------
    model : dict
        A fitted dual-SPLS model containing keys 'Bhat' and 'intercept'.
    X_new : np.ndarray
        New predictor matrix.
    
    Returns
    -------
    np.ndarray
        Predicted responses (for each component).
    """
    Bhat = model['Bhat']
    intercept = model['intercept']
    # For each component, compute X_new @ Bhat[:, comp] + intercept[comp]
    ncomp = Bhat.shape[1]
    predictions = np.zeros((X_new.shape[0], ncomp))
    for comp in range(ncomp):
        predictions[:, comp] = X_new @ Bhat[:, comp] + intercept[comp]
    return predictions
