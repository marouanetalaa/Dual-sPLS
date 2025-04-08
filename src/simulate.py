import numpy as np

def d_spls_simulate(n=200, p=100, nondes=50, sigmaondes=0.05, sigmay=0.5, int_coef=None):
    if int_coef is None:
        int_coef = np.arange(1, 6)

    # Vérification des dimensions de nondes et p
    if isinstance(p, int):
        p = [p]
    if len(nondes) != len(p):
        if len(nondes) == 1:
            print("Warning: nondes is only specified for the first mixture. Same value is considered for the rest.")
            nondes = [nondes[0]] * len(p)
        else:
            raise ValueError("Dimensions of nondes and p differ.")

    if len(sigmaondes) != len(p):
        if len(sigmaondes) == 1:
            print("Warning: sigmaondes is only specified for the first mixture. Same value is considered for the rest.")
            sigmaondes = [sigmaondes[0]] * len(p)
        else:
            raise ValueError("Dimensions of sigmaondes and p differ.")

    # Initialisation de la matrice des prédicteurs X
    X = np.zeros((n, sum(p)))

    # Générer les premières matrices de mélange
    ampl = np.random.rand(nondes[0], n)  # Amplitude de chaque Gaussienne
    modes = np.random.rand(nondes[0])  # Moyenne des Gaussiennes
    xech = np.linspace(0, 1, p[0])  # Variables pour la première matrice

    # Calcul de la première matrice de mélange
    for j in range(n):
        for io in range(nondes[0]):
            X[j, :p[0]] += ampl[io, j] * np.exp(-(xech - modes[io])**2 / (2 * sigmaondes[0]**2))

    # Calcul des autres matrices si p > 1
    if len(p) > 1:
        start_idx = p[0]
        for i in range(1, len(p)):
            ampl = np.random.rand(nondes[i], n)  # Amplitude de chaque Gaussienne
            modes = np.random.rand(nondes[i])  # Moyenne des Gaussiennes
            xech = np.linspace(0, 1, p[i])  # Variables pour la matrice suivante
            for j in range(n):
                for io in range(nondes[i]):
                    X[j, start_idx:start_idx + p[i]] += ampl[io, j] * np.exp(-(xech - modes[io])**2 / (2 * sigmaondes[i]**2))
            start_idx += p[i]

    # Calcul de y0 (réponse sans bruit)
    y0 = np.zeros(n)
    pif = np.round(np.linspace(10, 100, len(int_coef)) * sum(p) / 100).astype(int)
    pif = np.insert(pif, 0, 0)

    sumX = np.zeros((n, len(int_coef)))
    for i in range(len(int_coef)):
        sumX[:, i] = np.sum(X[:, pif[i]:pif[i+1]], axis=1)
    
    y0 = sumX @ int_coef

    # Ajout du bruit à y0 pour obtenir y
    y = y0 + sigmay * np.random.randn(n)
    y = np.vectorize(float)(y)  # Assure que y est un vecteur de type float
    G = len(p)

    return {
        'X': X,
        'y': y,
        'y0': y0,
        'sigmay': sigmay,
        'sigmaondes': sigmaondes,
        'G': G
    }

# Exemple d'utilisation
data = d_spls_simulate(n=100, p=[50, 100], nondes=[20, 30], sigmaondes=[0.05, 0.02], sigmay=0.5)
print(data['X'].shape)
print(data['y'].shape)
