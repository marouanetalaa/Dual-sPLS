import numpy as np
from numpy.linalg import svd, inv
from scipy.linalg import lstsq

def d_spls_LS(X, y, ncp, ppnu, verbose=True):
    n, p = X.shape  # Dimensions de X

    # Centrage des données
    Xm = np.mean(X, axis=0)
    Xc = X - Xm
    ym = np.mean(y)
    yc = y - ym

    # Initialisation des matrices
    WW = np.zeros((p, ncp))  # Matrice des poids
    TT = np.zeros((n, ncp))  # Matrice des scores
    Bhat = np.zeros((p, ncp))  # Matrice des coefficients
    YY = np.zeros((n, ncp))  # Matrice des réponses estimées
    RES = np.zeros((n, ncp))  # Matrice des résidus
    intercept = np.zeros(ncp)  # Intercept
    zerovar = np.zeros(ncp)  # Nombre de coefficients nuls par composant
    listelambda = np.zeros(ncp)  # Liste des valeurs de lambda
    ind_diff0 = [[] for _ in range(ncp)]  # Liste des indices des coefficients non nuls

    Xi = Xc  # Initialisation de X pour la déflation

    for ic in range(ncp):
        # Calcul de zi
        zi = np.dot(Xi.T, yc)
        
        # Calcul de la décomposition en valeurs singulières de Xi
        U, S, Vt = svd(Xi, full_matrices=False)
        invD = 1 / S
        if ic == 0 and min(S) / max(S) < 1e-16:
            print('XtX est proche de la singularité')
        elif ic > 0 and min(S[-(ncp - ic + 2):]) / max(S) < 1e-16:
            print(f'XtX défléchi est proche de la singularité pour le composant {ic}')
            invD[-(ncp - ic + 2):] = 0
        XtXmoins1 = np.dot(Vt.T, np.diag(invD**2).dot(Vt))

        # Optimisation de nu
        wLS = np.dot(XtXmoins1, zi)
        wLSs = np.sort(np.abs(wLS))
        wsp = np.arange(1, p + 1) / p
        iz = np.argmin(np.abs(wsp - ppnu))
        delta = np.sign(np.dot(XtXmoins1, zi))

        # Calcul de nu
        nu = wLSs[iz]

        # Calcul de znu
        znu = np.sign(wLS) * np.maximum(np.abs(wLS) - nu, 0)
        XZnu = np.dot(Xi, znu)
        XZnu2 = np.linalg.norm(XZnu, axis=0)
        mu = XZnu2
        lambda_ = nu / mu

        # Calcul de w et t
        w = znu
        WW[:, ic] = w

        # Calcul de t
        t = np.dot(Xi, w)
        t /= np.linalg.norm(t)
        TT[:, ic] = t

        # Déflation
        Xi -= np.outer(t, np.dot(t.T, Xi))

        # Calcul des coefficients Bhat
        R = np.dot(TT[:, :ic], Xc).dot(WW[:, :ic].T)
        np.fill_diagonal(R, 0)  # Stabilité numérique
        L = lstsq(R, np.eye(ic))[0]
        Bhat[:, ic] = np.dot(WW[:, :ic], np.dot(L, np.dot(TT[:, :ic].T, yc)))

        # Calcul de lambda
        listelambda[ic] = lambda_

        # Calcul de l'intercept
        intercept[ic] = ym - np.dot(Xm, Bhat[:, ic])

        # Prédictions
        YY[:, ic] = np.dot(X, Bhat[:, ic]) + intercept[ic]

        # Résidus
        RES[:, ic] = y - YY[:, ic]

        # Nombre de coefficients nuls
        zerovar[ic] = np.sum(Bhat[:, ic] == 0)

        # Indices des coefficients non nuls
        ind_diff0[ic] = np.where(Bhat[:, ic] != 0)[0]

        # Affichage des résultats
        if verbose:
            print(f'Dual PLS LS, ic={ic}, nu={nu}, nbzeros={zerovar[ic]}')

    return {
        'Xmean': Xm,
        'scores': TT,
        'loadings': WW,
        'Bhat': Bhat,
        'intercept': intercept,
        'fitted_values': YY,
        'residuals': RES,
        'lambda': listelambda,
        'zerovar': zerovar,
        'ind_diff0': ind_diff0,
        'type': "LS"
    }

## test

import unittest

class TestDSPLS(unittest.TestCase):

    def test_d_spls_LS(self):
        X = np.loadtxt('matrixXNirSpectrumData.csv', delimiter=',').T
        print(X.shape)
        y = np.loadtxt('matrixYNirPropertyDensityNormalized.csv', delimiter=',')##.reshape(-1, 1)
        print(y.shape)
        n, p = X.shape

        # Paramètres du modèle
        ncp = 5
        ppnu = 0.9

        # Ajuster le modèle
        mod_dspls = d_spls_LS(X=X, y=y, ncp=ncp, ppnu=ppnu, verbose=True)

        # Vérification des dimensions
        self.assertEqual(mod_dspls['scores'].shape, (n, ncp))
        self.assertEqual(len(mod_dspls['intercept']), ncp)
        self.assertEqual(mod_dspls['Bhat'].shape, (p, ncp))
        self.assertEqual(mod_dspls['loadings'].shape, (p, ncp))
        self.assertEqual(mod_dspls['fitted_values'].shape, (n, ncp))

        # Vérification des résidus
        np.testing.assert_array_equal(mod_dspls['residuals'], y - mod_dspls['fitted_values'])

        # Vérification de la moyenne de X
        np.testing.assert_array_equal(np.mean(X, axis=0), mod_dspls['Xmean'])

        # Vérification de zerovar
        for i in range(1, ncp):
            self.assertGreater(mod_dspls['zerovar'][i - 1], mod_dspls['zerovar'][i] - 1)
        self.assertLess(mod_dspls['zerovar'][0], ppnu * p + 1)

if __name__ == '__main__':
    unittest.main()
