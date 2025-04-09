import numpy as np

def mm_dual_norm_elasticnet(v, lambda1, lambda2, rho=10.0, tol=1e-6, max_iter=100, eps=1e-8, ppnu=0.8):
    """
    Compute the dual update for the elastic net penalty using an MM algorithm
    with a barrier and sparsity control via ppnu.
    
    We wish to solve:
        max_{w} <v, w>    subject to    lambda1 * ||w||_1 + lambda2 * ||w||_2^2 <= 1.
    Equivalently, we minimize:
        L(w) = - v^T w - rho * log(1 - [lambda1 * ||w||_1 + lambda2 * ||w||_2^2])
    
    To introduce sparsity, we first threshold v such that approximately
    a proportion ppnu of its coordinates are shrunk (set to zero).
    
    Parameters:
      v : numpy.ndarray of shape (p,)
          Input vector (e.g. z = Xdef.T @ yc) used in the dual problem.
      lambda1 : float
          L1 penalty parameter.
      lambda2 : float
          L2 penalty parameter.
      rho : float
          Barrier parameter.
      tol : float
          Tolerance for convergence of the MM iterations.
      max_iter : int
          Maximum number of MM iterations.
      eps : float
          Small constant for numerical stability.
      ppnu : float (between 0 and 1)
          Proportion of variables to be shrunk (promotes sparsity).
    
    Returns:
      w : numpy.ndarray of shape (p,)
          The computed weight vector.
      dual_val : float
          The dual norm value (v^T w).
    """
    p = v.shape[0]
    
    # --- Step 1. Introduce Sparsity via Thresholding ---
    abs_v = np.abs(v)
    sorted_abs = np.sort(abs_v)
    # Determine threshold nu from the empirical CDF of |v|
    index_thresh = int(np.floor(ppnu * p))
    index_thresh = min(max(index_thresh, 0), p-1)
    nu = sorted_abs[index_thresh]
    # Soft-threshold v: shrink any coordinate below nu to zero.
    v_thresh = np.sign(v) * np.maximum(np.abs(v) - nu, 0)
    # Now v is the thresholded vector used in subsequent updates.
    v = v_thresh

    # --- Step 2. Initialize the MM algorithm ---
    # We start with an initial guess for w (nonzero to prevent divide-by-zero)
    w = np.ones(p) * 0.1
    
    for _ in range(max_iter):
        # Compute current activity A = lambda1 * ||w||_1 + lambda2 * ||w||_2^2
        A = lambda1 * np.sum(np.abs(w)) + lambda2 * np.sum(w**2)
        if A >= 1:
            # If infeasible, scale down w
            w *= 0.99 / (A + eps)
            A = lambda1 * np.sum(np.abs(w)) + lambda2 * np.sum(w**2)
        
        upsilon = 1 - A  # feasibility gap
        
        # Build a surrogate for the barrier term.
        # For the L1 part, use a quadratic approximation:
        #   |w_i| ~ w_i^2 / (|w_i| + eps)  (up to constant)
        diag_entries = 1.0 / (np.abs(w) + eps)  # this gives a weight for each coordinate
        # Form the effective quadratic matrix for the penalty:
        Q = lambda1 * np.diag(diag_entries) + lambda2 * np.eye(p)
        
        # The surrogate objective (ignoring constant terms) is:
        #   g(w) = -v^T w + [rho/(1-A)] * w^T Q w.
        # Setting the derivative with respect to w to zero leads to:
        #   Q w_new = (1-A)/(2*rho) * v.
        try:
            w_new = (upsilon / (2 * rho)) * np.linalg.solve(Q, v)
        except np.linalg.LinAlgError:
            w_new = (upsilon / (2 * rho)) * np.linalg.lstsq(Q, v, rcond=None)[0]
        
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
        
    dual_val = np.dot(v, w)
    return w, dual_val

def d_spls_elasticnet_mm(X, y, ncp, lambda1=0.5, lambda2=0.1, rho=10.0, ppnu=0.8, verbose=True):
    """
    Dual-sPLS Elastic Net using an MM algorithm to compute the dual norm update.
    
    The elastic net penalty is defined as:
         Omega(w) = lambda1 * ||w||_1 + lambda2 * ||w||_2^2,
    and we seek a weight vector w maximizing <z, w> subject to Omega(w) <= 1.
    
    Sparsity is introduced by thresholding z according to ppnu.
    
    Parameters:
      X : numpy.ndarray of shape (n_samples, n_features)
      y : numpy.ndarray of shape (n_samples,)
      ncp : int
          Number of latent components.
      lambda1 : float
          L1 penalty weight.
      lambda2 : float
          L2 penalty weight.
      rho : float
          Barrier parameter for the MM algorithm.
      ppnu : float (in [0,1])
          Proportion of variables to shrink (controls sparsity).
      verbose : bool
          Whether to print iteration details.
    
    Returns:
      A dictionary with keys: 'Xmean', 'scores', 'loadings', 'Bhat',
      'intercept', 'fitted_values', 'residuals', 'dualvals', 'zerovar',
      'ind_diff0', and 'type'.
    """
    X = np.array(X)
    y = np.ravel(y)
    n, p = X.shape

    # Centering
    Xm = np.mean(X, axis=0)
    Xc = X - Xm
    ym = np.mean(y)
    yc = y - ym

    # Initialize matrices for loadings, scores, regression coefficients, etc.
    WW = np.zeros((p, ncp))
    TT = np.zeros((n, ncp))
    Bhat = np.zeros((p, ncp))
    YY = np.zeros((n, ncp))
    RES = np.zeros((n, ncp))
    intercept = np.zeros(ncp)
    zerovar = np.zeros(ncp, dtype=int)
    dualvals = np.zeros(ncp)
    ind_diff0 = [None] * ncp

    # Copy of Xc for deflation
    Xdef = Xc.copy()

    for ic in range(ncp):
        # Compute z from the current deflated matrix
        z = Xdef.T @ yc
        
        # Use the MM algorithm with sparsity (ppnu) to obtain the weight vector and dual value.
        w, dualval = mm_dual_norm_elasticnet(z, lambda1, lambda2, rho=rho, tol=1e-6,
                                               max_iter=100, eps=1e-8, ppnu=ppnu)
        WW[:, ic] = w
        dualvals[ic] = dualval

        # Compute score vector t = Xdef @ w and normalize
        t_vec = Xdef @ w
        norm_t = np.linalg.norm(t_vec, 2)
        t_vec_norm = t_vec if norm_t == 0 else t_vec / norm_t
        TT[:, ic] = t_vec_norm

        # Deflation: remove the projection of Xdef onto t_vec_norm
        Xdef = Xdef - np.outer(t_vec_norm, t_vec_norm) @ Xdef

        # Compute regression coefficients using all components so far.
        TT_current = TT[:, :ic+1]
        WW_current = WW[:, :ic+1]
        R = (TT_current.T @ Xc) @ WW_current
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if i > j:
                    R[i, j] = 0
        try:
            L = np.linalg.solve(R, np.eye(ic+1))
        except np.linalg.LinAlgError:
            L = np.linalg.lstsq(R, np.eye(ic+1), rcond=None)[0]
        Bhat[:, ic] = WW_current @ (L @ (TT_current.T @ yc))

        intercept[ic] = ym - Xm @ Bhat[:, ic]
        zerovar[ic] = np.sum(Bhat[:, ic] == 0)
        ind_diff0[ic] = np.where(Bhat[:, ic] != 0)[0].tolist()

        YY[:, ic] = X @ Bhat[:, ic] + intercept[ic]
        RES[:, ic] = y - YY[:, ic]
        if verbose:
            print(f"[MM Dual-sPLS EN] Component {ic+1}, Dual norm = {dualval:.4f}")
            
    return {
        'Xmean': Xm,
        'scores': TT,
        'loadings': WW,
        'Bhat': Bhat,
        'intercept': intercept,
        'fitted_values': YY,
        'residuals': RES,
        'dualvals': dualvals,
        'zerovar': zerovar,
        'ind_diff0': ind_diff0,
        'type': 'elasticnet-mm'
    }

