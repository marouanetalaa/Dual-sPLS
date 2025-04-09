import numpy as np
import itertools
from scipy.linalg import solve_triangular

def d_spls_norm1(v):
    """Renvoie la norme L1 (somme des valeurs absolues)."""
    return np.sum(np.abs(v))

def d_spls_norm2(v):
    """Renvoie la norme L2 (norme Euclidienne)."""
    return np.linalg.norm(v)

def d_spls_GLA(X, y, ncp, ppnu, indG, verbose=False):
    """
    Dual Sparse Partial Least Squares (Dual-SPLS) regression with group lasso norm type "GLA".
    
    Paramètres
    ----------
    X : ndarray, shape (n, p)
        Matrice des prédicteurs. Chaque ligne représente une observation,
        chaque colonne une variable.
    y : ndarray, shape (n,) ou (n,1)
        Vecteur (ou matrice colonne) de la réponse.
    ncp : int
        Nombre de composantes Dual-SPLS.
    ppnu : float or ndarray
        Proportion désirée (ou vecteur de proportions, de longueur égale au nombre de groupes)
        de variables à mettre à zéro dans chaque groupe.
    indG : ndarray
        Vecteur contenant, pour chaque variable (colonne de X), l'indice du groupe auquel elle appartient.
        (Les indices de groupe doivent commencer à 1.)
    verbose : bool, par défaut False
        Si True, affiche des informations sur les itérations.
        
    Renvoie
    -------
    result : dict
        Dictionnaire contenant les éléments suivants :
            - "Xmean": vecteur des moyennes des colonnes de X.
            - "scores": matrice (n, ncp) des scores.
            - "loadings": matrice (p, ncp) des vecteurs de chargement.
            - "Bhat": matrice (p, ncp) des coefficients de régression.
            - "intercept": vecteur (ncp,) des intercepts.
            - "fitted.values": matrice (n, ncp) des valeurs prédites.
            - "residuals": matrice (n, ncp) des résidus.
            - "lambda": matrice (nG, ncp) des paramètres de sparsité par groupe.
            - "alpha": matrice (nG, ncp) des paramètres de contraintes par groupe.
            - "zerovar": matrice (nG, ncp) du nombre de coefficients nuls par groupe.
            - "PP": vecteur indiquant le nombre de variables par groupe.
            - "ind.diff0": liste (de longueur ncp) des indices des coefficients non nuls pour chaque composante.
            - "type": chaîne indiquant le type de norme utilisé ("GLA").
    """
    # Assurer que X et y sont des tableaux NumPy
    X = np.asarray(X)
    y = np.asarray(y).flatten()  # on force à être un vecteur 1D
    
    n = len(y)         # nombre d'observations
    p = X.shape[1]     # nombre de variables

    # Centrage des données
    Xm = np.mean(X, axis=0)   # moyenne de X (pour chaque variable)
    Xc = X - Xm               # centrage de X
    ym = np.mean(y)
    yc = y - ym

    # Initialisation
    # Nombre de groupes (les indices contenus dans indG commencent à 1)
    indG = np.asarray(indG)
    nG = int(np.max(indG))
    PP = np.array([np.sum(indG == g) for g in range(1, nG+1)])
    
    # Initialisation des matrices
    WW = np.zeros((p, ncp))          # matrice des loadings
    TT = np.zeros((n, ncp))          # matrice des scores
    Bhat = np.zeros((p, ncp))        # coefficients de régression
    YY = np.zeros((n, ncp))          # valeurs prédites
    RES = np.zeros((n, ncp))         # résidus
    intercept = np.zeros(ncp)        # intercepts
    zerovar = np.zeros((nG, ncp), dtype=int)   # nombres de coefficients nuls par groupe
    listelambda = np.zeros((nG, ncp))
    listealpha = np.zeros((nG, ncp))
    ind_diff0 = [None] * ncp         # liste des indices de coefficients non nuls par composante
    
    # Variables intermédiaires
    nu = np.zeros(nG)                # seuil nu par groupe
    Znu = np.zeros(p)                # vecteur Znu global
    w = np.zeros(p)                  # vecteur de poids (sera reconstruit pour chaque groupe)
    norm2Znu = np.zeros(nG)          # norme L2 de Znu pour chaque groupe
    norm1Znu = np.zeros(nG)          # norme L1 de Znu pour chaque groupe

    # Préparation de la déflation
    Xdef = Xc.copy()   # X défait, qui sera mis à jour à chaque composante

    # Si ppnu est un scalaire, on l'étend à tous les groupes
    if np.isscalar(ppnu):
        ppnu = np.repeat(ppnu, nG)
    else:
        ppnu = np.asarray(ppnu)
        if ppnu.size != nG:
            raise ValueError("La longueur de ppnu doit être égale au nombre de groupes (max(indG)).")

    # Boucle sur les composantes
    for ic in range(ncp):
        # Calcul du vecteur de corrélations
        Z = Xdef.T.dot(yc)
        
        # Pour chaque groupe, optimisation du seuil nu et calcul de Znu
        for ig in range(1, nG+1):
            # Indices des variables appartenant au groupe ig
            idx = np.where(indG == ig)[0]
            # Trie des valeurs absolues de Z pour ces variables
            Z_abs_sorted = np.sort(np.abs(Z[idx]))
            d_val = len(Z_abs_sorted)
            # Création d'un vecteur échelonné (valeurs allant de 1/d à 1)
            Zsp = np.arange(1, d_val+1) / d_val
            # Trouver l'indice minimisant la différence absolue avec ppnu[ig]
            diff = np.abs(Zsp - ppnu[ig-1])
            iz = np.argmin(diff)
            nu[ig-1] = Z_abs_sorted[iz]
            # Calcul de Znu pour le groupe ig
            # Pour chaque u dans Z[idx]: applique sign(u)*max(|u|-nu, 0)
            Znu[idx] = np.sign(Z[idx]) * np.maximum(np.abs(Z[idx]) - nu[ig-1], 0)
            # Calcul des normes
            norm1Znu[ig-1] = d_spls_norm1(Znu[idx])
            norm2Znu[ig-1] = d_spls_norm2(Znu[idx])
        
        # Calcul de mu (somme des normes L2 de chaque groupe)
        mu = np.sum(norm2Znu)
        # Calcul de alpha et lambda pour chaque groupe
        alpha = norm2Znu / mu
        lam = nu / (mu * alpha)   # lambda par groupe
        
        # Calcul de la valeur maximale de la norme L2 pour chaque groupe
        max_norm2w = np.array([1/alpha[g] / (1 + (nu[g]*norm1Znu[g])/(mu*alpha[g])**2)
                                 for g in range(nG)])
        
        # Construction de la grille pour les poids pour les premiers nG-1 groupes
        grid_values = [np.linspace(0, max_norm2w[g], 10) for g in range(nG-1)]
        # Toutes les combinaisons possibles de valeurs pour les groupes 1 à nG-1
        grid_tuples = list(itertools.product(*grid_values))
        grid_arr = np.array(grid_tuples)  # forme (ncomb, nG-1)
        ncomb = grid_arr.shape[0]
        # On initialisera la matrice "comb" en ajoutant une colonne pour le groupe nG
        comb = np.zeros((ncomb, nG))
        comb[:, :nG-1] = grid_arr
        
        # Calcul de la dernière colonne pour satisfaire la contrainte linéaire
        denom = alpha[nG-1] * (1 + (nu[nG-1]*norm1Znu[nG-1])/(mu*alpha[nG-1])**2)
        for i in range(ncomb):
            num = 0
            for u in range(nG-1):
                num += alpha[u] * comb[i, u] * (1 + (nu[u]*norm1Znu[u])/(mu*alpha[u])**2)
            num = 1 - num
            comb[i, nG-1] = num / denom
        
        # Suppression des lignes inadéquates (pour lesquelles la dernière valeur est négative)
        mask = comb[:, nG-1] >= 0
        comb = comb[mask]
        ncomb = comb.shape[0]
        
        # Pour chaque combinaison, on calcule la solution candidate et l'erreur quadratique moyenne (RMSE)
        RMSE = np.zeros(ncomb)
        tempw = np.zeros((p, ncomb))
        
        for icomb in range(ncomb):
            # Pour chaque groupe, reconstruire le vecteur w
            w_candidate = np.zeros(p)
            for ig in range(1, nG+1):
                idx = np.where(indG == ig)[0]
                # Pour le groupe ig, la valeur prise dans la combinaison (attention aux indices : en Python, 0-indexé)
                wval = comb[icomb, ig-1]
                w_candidate[idx] = (wval / (mu * alpha[ig-1])) * Znu[idx]
            
            # Calcul du vecteur score t candidate
            t_candidate = Xdef.dot(w_candidate)
            norm_t = d_spls_norm2(t_candidate)
            # Pour éviter une division par zéro
            if norm_t == 0:
                continue
            t_candidate = t_candidate / norm_t
            
            # Constitution des matrices temporaires incluant les composantes déjà fixées
            # On concatène les composantes précédentes (le cas échéant) et la composante candidate actuelle.
            if ic == 0:
                TT_candidate = t_candidate.reshape(-1, 1)
                WW_candidate = w_candidate.reshape(-1, 1)
            else:
                TT_candidate = np.hstack((TT[:, :ic], t_candidate.reshape(-1, 1)))
                WW_candidate = np.hstack((WW[:, :ic], w_candidate.reshape(-1, 1)))
            
            # Calcul de R = (TT_candidate)^T * Xc * WW_candidate
            R = TT_candidate.T.dot(Xc.dot(WW_candidate))
            # Pour stabilité numérique : ne conserver que la partie triangulaire supérieure
            R = np.triu(R)
            # Résolution du système triangulaire R * L = I
            try:
                L = solve_triangular(R, np.eye(R.shape[0]), lower=False)
            except np.linalg.LinAlgError:
                # Si le système est mal conditionné, on passe à la combinaison suivante
                RMSE[icomb] = np.inf
                continue
            
            # Calcul de Bhat candidate pour la composante ic (seule la dernière colonne est « active »)
            Bhat_candidate = WW_candidate.dot(L.dot(TT_candidate.T.dot(yc)))
            
            # Calcul de l'intercept et des valeurs prédites
            intercept_candidate = ym - Xm.dot(Bhat_candidate)
            Y_candidate = X.dot(Bhat_candidate) + intercept_candidate
            residuals_candidate = y - Y_candidate
            RMSE[icomb] = np.sum(residuals_candidate**2) / n
            
            # Enregistrement du vecteur w candidate
            tempw[:, icomb] = w_candidate
        
        # Choix de la solution candidate optimale (celle minimisant le RMSE)
        if np.all(np.isinf(RMSE)):
            raise RuntimeError("Aucune combinaison candidate ne donne une solution admissible.")
        icomb_opt = np.argmin(RMSE)
        w = tempw[:, icomb_opt].copy()
        WW[:, ic] = w  # mise à jour du loading pour la composante courante
        
        # Recalcule du vecteur score t associé à la solution optimale
        t = Xdef.dot(w)
        norm_t = d_spls_norm2(t)
        if norm_t == 0:
            raise RuntimeError("Norme nulle rencontrée lors du calcul du score.")
        t = t / norm_t
        TT[:, ic] = t
        
        # Déflation : suppression de la contribution de la composante courante
        # Xdef = Xdef - t * (t^T * Xdef)
        Xdef = Xdef - np.outer(t, t.dot(Xdef))
        
        # Recalcul des coefficients à partir des composantes obtenues jusque-là
        TT_current = TT[:, :ic+1]
        WW_current = WW[:, :ic+1]
        R = TT_current.T.dot(Xc.dot(WW_current))
        R = np.triu(R)
        L = solve_triangular(R, np.eye(R.shape[0]), lower=False)
        Bhat_current = WW_current.dot(L.dot(TT_current.T.dot(yc)))
        Bhat[:, ic] = Bhat_current
        
        # Enregistrement des paramètres lambda et alpha pour la composante actuelle
        listelambda[:, ic] = lam
        listealpha[:, ic] = alpha
        
        # Calcul de l'intercept, des prédictions et des résidus pour cette composante
        intercept[ic] = ym - Xm.dot(Bhat[:, ic])
        YY[:, ic] = X.dot(Bhat[:, ic]) + intercept[ic]
        RES[:, ic] = y - YY[:, ic]
        
        # Calcul du nombre de coefficients nuls par groupe pour la composante ic
        for ig in range(1, nG+1):
            idx = np.where(indG == ig)[0]
            zerovar[ig-1, ic] = np.sum(Bhat[idx, ic] == 0)
        
        # Indices des coefficients non nuls pour la composante ic
        ind_diff0[ic] = np.where(Bhat[:, ic] != 0)[0]
        
        if verbose:
            print(f"Dual PLS ic = {ic+1} | lambda = {lam} | mu = {mu} | nu = {nu} | nbzeros = {zerovar[:, ic]}")
    
    # Constitution du résultat final dans un dictionnaire
    result = {
        "Xmean": Xm,
        "scores": TT,
        "loadings": WW,
        "Bhat": Bhat,
        "intercept": intercept,
        "fitted_values": YY,
        "residuals": RES,
        "lambda": listelambda,
        "alpha": listealpha,
        "zerovar": zerovar,
        "PP": PP,
        "ind.diff0": ind_diff0,
        "type": "GLA"
    }
    
    return result

# Exemple d'utilisation (à adapter selon vos données) :
if __name__ == "__main__":
    # Données fictives pour illustrer l'appel de la fonction
    np.random.seed(0)
    n, p = 100, 20
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    ncp = 3
    # Supposons que les 20 variables soient réparties en 4 groupes (indices de 1 à 4)
    indG = np.repeat(np.arange(1, 5), p/4)
    # ppnu peut être un scalaire (par exemple 0.3) ou un vecteur de longueur 4
    ppnu = 0.3
    
    res = d_spls_GLA(X, y, ncp, ppnu, indG, verbose=True)
    # Affichage des résultats
    print("Scores:", res["scores"])
    print("Loadings:", res["loadings"])
    print("Coefficients (Bhat):", res["Bhat"])
    print("Intercepts:", res["intercept"])
