
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from dual_spls import GLA
from dual_spls import GLB
from dual_spls import GLC


def cluster_variables_fixed_groups(X, method='ward', n_groups=3):

    # Calcul de la matrice de corrélation entre variables (colonnes)
    corr = np.corrcoef(X, rowvar=False)

    # Transformation de la corrélation en distance.
    # Par exemple, distance = 1 - |corrélation|
    distance = 1 - np.abs(corr)

    # Transformation de la matrice de distance en format condensé
    condensed_distance = squareform(distance, checks=False)
    
    # Clustering hiérarchique
    Z = linkage(condensed_distance, method=method)
    
    # Formation des clusters en fixant le nombre maximum de clusters
    indG = fcluster(Z, t=n_groups, criterion='maxclust')
    
    return indG

def d_spls_GL(X, y, ncp, ppnu, indG, gamma=None, norm="A", verbose=False):
    """
    Dual Sparse Partial Least Squares (Dual-SPLS) regression for the group lasso norms.

    Parameters
    ----------
    X : np.ndarray
        A numeric matrix of predictors values of shape (n_samples, n_features).
    y : np.ndarray
        A numeric vector or column matrix of responses.
    ncp : int
        The number of Dual-SPLS components.
    ppnu : float or list of float
        Proportion of variables to shrink to zero, per group.
    indG : list or np.ndarray
        Group index for each feature.
    gamma : list or np.ndarray, optional
        Required if norm="C". Weights for each group.
    norm : str, default="A"
        Chosen norm. One of "A", "B", or "C".
    verbose : bool, default=False
        Whether to print the iteration steps.

    Returns
    -------
    dict
        A dictionary with model components as keys:
        'Xmean', 'scores', 'loadings', 'Bhat', 'intercept',
        'fitted_values', 'residuals', 'lambda', 'alpha',
        'zerovar', 'PP', 'ind_diff0', 'type'
    """
    if norm == "A":
        mod_dspls = GLA.d_spls_GLA(X=X, y=y, ncp=ncp, ppnu=ppnu, indG=indG, verbose=verbose)
    elif norm == "B":
        mod_dspls = GLB.d_spls_GLB(X=X, y=y, ncp=ncp, ppnu=ppnu, indG=indG, verbose=verbose)
    elif norm == "C":
        mod_dspls = GLC.d_spls_GLC(X=X, y=y, ncp=ncp, ppnu=ppnu, indG=indG, gamma=gamma, verbose=verbose)
    else:
        raise ValueError("Invalid norm type. Choose 'A', 'B', or 'C'.")
    
    return mod_dspls



