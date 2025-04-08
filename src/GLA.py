import numpy as np
from scipy.linalg import lstsq

def d_spls_GLA(X, y, ncp, ppnu, indG, verbose=False):
    """
    Dual Sparse Partial Least Squares (Dual-SPLS) regression with group lasso norm.

    Parameters:
    X : numpy.ndarray
        A (n, p) matrix of predictor values, where each row is an observation and each column is a predictor variable.
    y : numpy.ndarray
        A (n,) response vector.
    ncp : int
        The number of Dual-SPLS components.
    ppnu : list or float
        A positive real value or a vector of length equal to the number of groups, representing the desired proportion of variables to shrink to zero for each component and each group.
    indG : numpy.ndarray
        A numeric vector of group indices for each observation.
    verbose : bool, optional
        Whether to display iteration steps. Default is False.

    Returns:
    dict
        A dictionary with the following results:
        - Xmean: Mean vector of X
        - scores: Matrix of scores
        - loadings: Matrix of loadings
        - Bhat: Matrix of regression coefficients
        - intercept: Intercept values for each component
        - fitted_values: Predicted values of y
        - residuals: Residuals (difference between responses and fitted values)
        - lambda_: Lambda values for each group
        - alpha: Alpha values for each group
        - zerovar: Number of variables shrunk to zero per component and group
        - PP: Number of variables in each group
        - ind_diff0: Indices of non-zero regression coefficients for each component
        - type: Specifies the Dual-SPLS norm used, in this case "GLA"
    """
    # Dimensions
    n = len(y)  # Number of observations
    p = X.shape[1]  # Number of predictor variables

    # Centering Data
    Xm = np.mean(X, axis=0)  # Mean of X
    Xc = X - np.ones((n, 1)) @ Xm.reshape(1, p)  # Centering X
    ym = np.mean(y)  # Mean of y
    yc = y - ym  # Centering y

    # Initialization
    nG =int( np.max(indG))  # Number of groups
    PP = np.array([np.sum(indG == g) for g in range(1, nG+1)])  # Number of variables in each group

    WW = np.zeros((p, ncp))  # Matrix of loadings
    TT = np.zeros((n, ncp))  # Matrix of scores
    Bhat = np.zeros((p, ncp))  # Matrix of coefficients
    YY = np.zeros((n, ncp))  # Predicted values
    RES = np.zeros((n, ncp))  # Residuals
    intercept = np.zeros(ncp)  # Intercept values
    zerovar = np.zeros((nG, ncp))  # Number of zero coefficients per group per component
    listelambda = np.zeros((nG, ncp))  # Lambda values
    listealpha = np.zeros((nG, ncp))  # Alpha values
    ind_diff0 = {f'in.diff0_{i+1}': [] for i in range(ncp)}  # Non-zero coefficient indices
    nu = np.zeros(nG)  # Nu for each group
    Znu = np.zeros(p)  # Znu for each group
    w = np.zeros(p)  # Weight vector
    norm2Znu = np.zeros(nG)  # Norm2 of Znu for each group
    norm1Znu = np.zeros(nG)  # Norm1 of Znu for each group

    # Dua-SPLS iteration
    Xdef = Xc  # Initializing X for deflation step
    for ic in range(ncp):
        Z = Xdef.T @ yc  # Calculate Z
        Z = Z.flatten()

        for ig in range(nG):
            # Index of the group
            ind = np.where(indG == (ig + 1))[0]

            # Optimize nu(g)
            Zs = np.sort(np.abs(Z[ind]))
            d = len(Zs)
            Zsp = np.arange(1, d + 1) / d
            iz = np.argmin(np.abs(Zsp - ppnu[ig]))
            nu[ig] = Zs[iz]

            # Finding Znu
            Znu[ind] = np.sign(Z[ind]) * np.maximum(np.abs(Z[ind]) - nu[ig], 0)

            # Norm 1 and 2 of Znu(g)
            norm1Znu[ig] = np.linalg.norm(Znu[ind], ord=1)
            norm2Znu[ig] = np.linalg.norm(Znu[ind], ord=2)

        # Compute alpha and lambda
        mu = np.sum(norm2Znu)
        alpha = norm2Znu / mu
        lambda_ = nu / (mu * alpha)

        # Max norm2 of wg
        max_norm2w = 1 / alpha / (1 + (nu * norm1Znu / (mu * alpha)**2))

        # Sample possible values of wg
        sample_wg = np.zeros((10, nG - 1))
        for ig in range(nG - 1):
            sample_wg[:, ig] = np.linspace(0, max_norm2w[ig], 10)

        # All possible combinations
        comb = np.array(np.meshgrid(*[sample_wg[:, ig] for ig in range(nG - 1)])).T.reshape(-1, nG - 1)
        comb = np.hstack([comb, np.zeros((comb.shape[0], 1))])

        denom = alpha[-1] * (1 + (nu[-1] * norm1Znu[-1] / (mu * alpha[-1])**2))

        for icomb in range(comb.shape[0]):
            numg = np.array([alpha[ig] * comb[icomb, ig] * (1 + (nu[ig] * norm1Znu[ig] / (mu * alpha[ig])**2)) for ig in range(nG - 1)])
            num = 1 - np.sum(numg)
            comb[icomb, nG - 1] = num / denom

        # Remove inadequate rows
        comb = comb[comb[:, nG - 1] >= 0]

        # Initialize RMSE
        RMSE = np.zeros(comb.shape[0])
        tempw = np.zeros((p, comb.shape[0]))

        # Compute w for each combination
        for icomb in range(comb.shape[0]):
            for ig in range(nG):
                ind = np.where(indG == (ig + 1))[0]
                w[ind] = (comb[icomb, ig] / (mu * alpha[ig])) * Znu[ind]

            # Compute T
            t = Xdef @ w
            t = t / np.linalg.norm(t, ord=2)

            WW[:, ic] = w
            TT[:, ic] = t

            # Coefficient vectors
            R = TT[:, :ic + 1].T @ Xc @ WW[:, :ic + 1]
            R[np.tril_indices(ic)] = 0  # Numerical stability
            L, _, _, _ = lstsq(R, np.eye(ic + 1))

            Bhat[:, ic] = WW[:, :ic + 1] @ (L @ (TT[:, :ic + 1].T @ yc))

            intercept[ic] = ym - Xm @ Bhat[:, ic]

            # Predictions
            YY[:, ic] = X @ Bhat[:, ic] + intercept[ic]
            RES[:, ic] = y - YY[:, ic]

            tempw[:, icomb] = w
            RMSE[icomb] = np.sum(RES[:, ic] ** 2) / n

        # Choosing optimal w
        indwmax = np.argmin(RMSE)
        w = tempw[:, indwmax]
        WW[:, ic] = w

        # Compute T
        t = Xdef @ w
        t = t / np.linalg.norm(t, ord=2)
        TT[:, ic] = t

        # Deflation
        Xdef = Xdef - t[:, None] @ t[None, :] @ Xdef

        # Coefficients
        R = TT[:, :ic + 1].T @ Xc @ WW[:, :ic + 1]
        R[np.tril_indices(ic)] = 0  # Numerical stability
        L, _, _, _ = lstsq(R, np.eye(ic + 1))

        Bhat[:, ic] = WW[:, :ic + 1] @ (L @ (TT[:, :ic + 1].T @ yc))

        # Store results
        listelambda[:, ic] = lambda_
        listealpha[:, ic] = alpha
        intercept[ic] = ym - Xm @ Bhat[:, ic]
        zerovar[:, ic] = np.array([np.sum(Bhat[np.where(indG == g), ic] == 0) for g in range(1, nG + 1)])
        ind_diff0[f'in.diff0_{ic + 1}'] = [np.where(Bhat[:, ic] != 0)[0]]

        # Predictions
        YY[:, ic] = X @ Bhat[:, ic] + intercept[ic]
        RES[:, ic] = y - YY[:, ic]

        if verbose:
            print(f"Dual PLS ic={ic}, lambda={lambda_}, mu={mu}, nu={nu}, nbzeros={zerovar[:, ic]}")

    return {
        'Xmean': Xm,
        'scores': TT,
        'loadings': WW,
        'Bhat': Bhat,
        'intercept': intercept,
        'fitted_values': YY,
        'residuals': RES,
        'lambda': listelambda,
        'alpha': listealpha,
        'zerovar': zerovar,
        'PP': PP,
        'ind_diff0': ind_diff0,
        'type': 'GLA'
    }




## Test
import unittest
import simulate
class TestDSPLS(unittest.TestCase):

    def test_d_spls_GLA(self):
        # Paramètres
        n = 100
        p = [50, 100]
        nondes = [20, 30]
        sigmaondes = [0.05, 0.02]

        # Simuler les données
        data = simulate.d_spls_simulate(n=n, p=p, nondes=nondes, sigmaondes=sigmaondes)
        X = data['X']
        y = data['y']

        # Diviser X en X1 et X2
        X1 = X[:, :p[0]]
        X2 = X[:, p[0]:p[1]]

        # Indices de groupe
        indG = np.concatenate([np.ones(p[0]), np.ones(p[1]) * 2])

        # Paramètres du modèle
        ncp = 10
        ppnu = [0.99, 0.9]

        # Ajuster le modèle
        mod_dspls = d_spls_GLA(X=X, y=y, ncp=ncp, ppnu=ppnu, indG=indG, verbose=True)
        n, p_val = X.shape

        # Vérification des dimensions
        self.assertEqual(mod_dspls['scores'].shape, (n, ncp))
        self.assertEqual(len(mod_dspls['intercept']), ncp)
        self.assertEqual(mod_dspls['Bhat'].shape, (p_val, ncp))
        self.assertEqual(mod_dspls['loadings'].shape, (p_val, ncp))
        self.assertEqual(mod_dspls['fitted_values'].shape, (n, ncp))

        # Vérification des résidus
        for i in range(ncp):
            np.testing.assert_array_equal(mod_dspls['residuals'][:,i], y - mod_dspls['fitted_values'][:,i])

        # Vérification de la moyenne de X
        np.testing.assert_array_equal(np.mean(X, axis=0), mod_dspls['Xmean'])

        # Vérification de zerovar
        for i in range(1, ncp):
            self.assertGreater(mod_dspls['zerovar'][0, i-1], mod_dspls['zerovar'][0, i] - 1)
        self.assertLess(mod_dspls['zerovar'][0, 0], ppnu[0] * p_val + 1)

        # Vérification du nombre de variables
        self.assertEqual(len(np.unique(indG)), mod_dspls['zerovar'].shape[0])

if __name__ == '__main__':
    unittest.main()
