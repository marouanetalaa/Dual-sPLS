import numpy as np

def d_spls_norm2(vec):

    return np.linalg.norm(vec, 2)

def d_spls_norm1(vec):

    return np.linalg.norm(vec, 1)

def d_spls_lasso(X, y, ncp, ppnu, verbose=True):

    # Conversion en tableaux NumPy et redimensionnement de y en vecteur 1D
    X = np.array(X)
    y = np.ravel(y)
    n = y.shape[0]
    p = X.shape[1]

    # Centrage des données
    Xm = np.mean(X, axis=0)           # Moyenne de chaque colonne de X
    Xc = X - Xm                     # Centrage de X (différence avec la moyenne)
    ym = np.mean(y)                 # Moyenne de y
    yc = y - ym                   # Centrage de y

    # Initialisations
    WW = np.zeros((p, ncp))         # Matrice des loadings (p x ncp)
    TT = np.zeros((n, ncp))         # Matrice des scores (n x ncp)
    Bhat = np.zeros((p, ncp))       # Matrice des coefficients de régression (p x ncp)
    YY = np.zeros((n, ncp))         # Matrice des valeurs prédites (n x ncp)
    RES = np.zeros((n, ncp))        # Matrice des résidus (n x ncp)
    intercept = np.zeros(ncp)       # Vecteur des intercepts pour chaque composante
    zerovar = np.zeros(ncp, dtype=int)   # Vecteur contenant le nombre de coefficients nuls par composante
    listelambda = np.zeros(ncp)     # Tableau des valeurs de lambda pour chaque composante
    ind_diff0 = [None] * ncp        # Liste pour stocker les indices des coefficients non nuls

    # Algorithme Dual-SPLS
    Xdef = Xc.copy()  # Copie de Xc pour la déflation
    for ic in range(ncp):
        # Calcul de z = Xdef^T yc
        Z = Xdef.T @ yc
        # Calcul de la valeur seuil nu :
        absZ = np.abs(Z)
        Zs = np.sort(absZ)                    # Tri croissant des valeurs absolues de Z
        Zsp = np.arange(1, p + 1) / p           # Vecteur allant de 1/p à 1
        iz = np.argmin(np.abs(Zsp - ppnu))      # Indice où |Zsp - ppnu| est minimal
        nu = Zs[iz]
        
        # Opérateur de seuillage doux : sign(Z) * max(|Z| - nu, 0)
        Znu = np.sign(Z) * np.maximum(np.abs(Z) - nu, 0)
        Znu2 = d_spls_norm2(Znu)
        Znu1 = d_spls_norm1(Znu)
        mu = Znu2
        lam = nu / mu if mu != 0 else 0

        # Calcul de w
        denominator = nu * Znu1 + mu**2
        if denominator == 0:
            w = np.zeros_like(Znu)
        else:
            w = (mu / denominator) * Znu

        WW[:, ic] = w

        # Calcul du vecteur score t et normalisation
        t_vec = Xdef @ w
        norm_t = d_spls_norm2(t_vec)
        if norm_t == 0:
            t_vec_normalized = t_vec
        else:
            t_vec_normalized = t_vec / norm_t
        TT[:, ic] = t_vec_normalized

        # Déflation : suppression de la projection t t^T * Xdef
        Xdef = Xdef - np.outer(t_vec_normalized, t_vec_normalized) @ Xdef

        # Calcul des coefficients de régression pour la composante ic
        # On considère les composantes calculées jusqu'à présent : TT_current (n x (ic+1)) et WW_current (p x (ic+1))
        TT_current = TT[:, :ic + 1]
        WW_current = WW[:, :ic + 1]
        R = (TT_current.T @ Xc) @ WW_current  # Matrice (ic+1 x ic+1)
        # Mise à zéro des éléments sous-diagonaux pour la stabilité numérique
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if i > j:
                    R[i, j] = 0

        # Résolution du système R * L = I (backsolve) pour obtenir L
        try:
            L = np.linalg.solve(R, np.eye(ic + 1))
        except np.linalg.LinAlgError:
            # En cas de matrice singulière, utiliser la résolution par moindres carrés
            L = np.linalg.lstsq(R, np.eye(ic + 1), rcond=None)[0]

        # Calcul du vecteur de coefficients pour la composante ic
        Bhat[:, ic] = WW_current @ (L @ (TT_current.T @ yc))

        listelambda[ic] = lam
        intercept[ic] = ym - Xm @ Bhat[:, ic]
        zerovar[ic] = np.sum(Bhat[:, ic] == 0)
        ind_diff0[ic] = np.where(Bhat[:, ic] != 0)[0].tolist()

        # Prédictions et résidus
        YY[:, ic] = X @ Bhat[:, ic] + intercept[ic]
        RES[:, ic] = y - YY[:, ic]

        if verbose:
            print(f'Dual PLS ic={ic+1}, lambda={lam:.4f}, mu={mu:.4f}, nu={nu:.4f}, nbzeros={zerovar[ic]}')

    # Rassembler les résultats dans un dictionnaire
    result = {
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
        'type': 'lasso'
    }
    return result