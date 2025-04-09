import numpy as np
from dual_spls import norm

def d_spls_GLC(X, y, ncp, ppnu, indG, gamma, verbose=False):
    if len(gamma) != len(np.unique(indG)):
        raise ValueError("incorrect length of gamma")

    if not np.isclose(np.sum(gamma), 1):
        raise ValueError("sum of gamma different than 1")

    n, p = X.shape
    Xmean = X.mean(axis=0)
    Xc = X - Xmean

    y = y.flatten() if y.ndim > 1 else y
    ymean = y.mean()
    yc = y - ymean

    nG = int(np.max(indG))
    PP = np.array([np.sum(indG == g) for g in range(1, nG+1)])

    WW = np.zeros((p, ncp))
    TT = np.zeros((n, ncp))
    Bhat = np.zeros((p, ncp))
    YY = np.zeros((n, ncp))
    RES = np.zeros((n, ncp))
    intercept = np.zeros(ncp)
    zerovar = np.zeros((nG, ncp))
    listelambda = np.zeros((nG, ncp))
    listealpha = np.zeros((nG, ncp))
    ind_diff0 = {}

    for ic in range(ncp):
        Z = Xc.T @ yc
        Z = Z.flatten()

        Znu = np.zeros(p)
        norm1Znu = np.zeros(nG)
        norm2Znu = np.zeros(nG)
        nu = np.zeros(nG)
        lambda_ = np.zeros(nG)
        alpha = np.zeros(nG)
        w = np.zeros(p)

        for ig in range(nG):
            ind = np.where(indG == ig + 1)[0]
            Zs = np.sort(np.abs(Z[ind]))
            d = len(Zs)
            Zsp = np.arange(1, d+1) / d
            iz = np.argmin(np.abs(Zsp - ppnu[ig]))
            nu[ig] = Zs[iz]

            Znu[ind] = np.sign(Z[ind]) * np.maximum(np.abs(Z[ind]) - nu[ig], 0)
            norm1Znu[ig] = norm.norm1(Znu[ind])
            norm2Znu[ig] = norm.norm2(Znu[ind])

        mu = np.sum(norm2Znu)

        for igg in range(nG):
            ind = np.where(indG == igg + 1)[0]
            alpha[igg] = norm2Znu[igg] / mu
            lambda_[igg] = nu[igg] / mu

            denom = alpha[igg] * norm2Znu[igg] + lambda_[igg] * norm1Znu[igg]
            w[ind] = (gamma[igg] * Znu[ind]) / denom if denom != 0 else 0

        WW[:, ic] = w
        t = Xc @ w
        t /= norm.norm2(t)
        TT[:, ic] = t

        Xc = Xc - np.outer(t, t @ Xc)

        R = TT[:, :ic+1].T @ (X - Xmean) @ WW[:, :ic+1]
        R = np.triu(R)
        L = np.linalg.solve(R.T, np.eye(ic+1)).T
        Bhat[:, ic] = WW[:, :ic+1] @ (L @ (TT[:, :ic+1].T @ yc))

        listelambda[:, ic] = lambda_
        listealpha[:, ic] = alpha
        intercept[ic] = ymean - Xmean @ Bhat[:, ic]

        for u in range(nG):
            ind_u = np.where(indG == u + 1)[0]
            zerovar[u, ic] = np.sum(Bhat[ind_u, ic] == 0)

        ind_diff0[f"in.diff0_{ic+1}"] = np.where(Bhat[:, ic] != 0)[0]

        YY[:, ic] = X @ Bhat[:, ic] + intercept[ic]
        RES[:, ic] = y - YY[:, ic]

        if verbose:
            print(f"Dual PLS ic={ic+1} lambda={lambda_}, mu={mu}, nu={nu}, nbzeros={zerovar[:, ic]}")

    return {
        'Xmean': Xmean,
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
        'type': 'GLC'
    }


import unittest
from dual_spls import simulate

class TestDSPLS(unittest.TestCase):

    def test_d_spls_GLC(self):
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
        gamma = [0.5, 0.5]

        # Ajuster le modèle
        mod_dspls = d_spls_GLC(X=X, y=y, ncp=ncp, ppnu=ppnu, indG=indG, gamma=gamma, verbose=True)
        n, p_val = X.shape

        # Vérification des dimensions
        self.assertEqual(mod_dspls['scores'].shape, (n, ncp))
        self.assertEqual(len(mod_dspls['intercept']), ncp)
        self.assertEqual(mod_dspls['Bhat'].shape, (p_val, ncp))
        self.assertEqual(mod_dspls['loadings'].shape, (p_val, ncp))
        self.assertEqual(mod_dspls['fitted_values'].shape, (n, ncp))

        # Vérification des résidus
        for i in range(ncp):
            np.testing.assert_array_equal(mod_dspls['residuals'][:, i], y - mod_dspls['fitted_values'][:, i])

        # Vérification de la moyenne de X
        np.testing.assert_array_equal(np.mean(X, axis=0), mod_dspls['Xmean'])

        # Vérification de zerovar pour chaque groupe
        for j in range(2):  # Pour les deux groupes
            for i in range(1, ncp):
                self.assertGreater(mod_dspls['zerovar'][j, i-1], mod_dspls['zerovar'][j, i] - 1)
            self.assertLess(mod_dspls['zerovar'][j, 0], ppnu[j] * p_val + 1)

        # Vérification du nombre de variables
        self.assertEqual(len(np.unique(indG)), mod_dspls['zerovar'].shape[0])

        # Vérification des erreurs
        with self.assertRaises(ValueError):
            d_spls_GLC(X=X, y=y, ncp=ncp, ppnu=ppnu, indG=indG, gamma=[0.5, 0.9], verbose=True)
        
        with self.assertRaises(ValueError):
            d_spls_GLC(X=X, y=y, ncp=ncp, ppnu=ppnu, indG=indG, gamma=[0.5], verbose=True)

if __name__ == '__main__':
    unittest.main()
