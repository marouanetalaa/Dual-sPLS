import numpy as np


# 1. Base class for norms
class BaseNorm:
    """
    Abstract base class for norm penalties.
    Subclasses must implement:
      - penalty(w): evaluates the norm penalty.
      - quadratic_approximation(w, eps): returns a quadratic surrogate matrix.
    """
    def __init__(self, **kwargs):
        # Flexible constructor for various norm parameters.
        pass

    def penalty(self, w):
        raise NotImplementedError("penalty() must be implemented by the subclass.")

    def quadratic_approximation(self, w, eps=1e-8):
        raise NotImplementedError("quadratic_approximation() must be implemented by the subclass.")


    def __repr__(self):
        return f"{self.__class__.__name__}"
# 2. LASSO norm class
##############################
class LassoNorm(BaseNorm):
    """
    Implements the LASSO (ℓ₁) norm:
         Omega(w) = lambda1 * ||w||₁.
    The quadratic surrogate for each coordinate is given by
         |w_i| ≈ w_i^2/(|w_i| + eps)
    to smooth the non-differentiability.
    """
    def __init__(self, lambda1):
        self.lambda1 = lambda1

    def penalty(self, w):
        return self.lambda1 * np.sum(np.abs(w))

    def quadratic_approximation(self, w, eps=1e-8):
        diag_entries = 1.0 / (np.abs(w) + eps)
        return self.lambda1 * np.diag(diag_entries)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(lambda1={self.lambda1})"



# 3. Elastic Net norm class

class ElasticNetNorm(BaseNorm):
    """
    Implements the Elastic Net norm:
         Omega(w) = lambda1 * ||w||₁ + lambda2 * ||w||₂².
    """
    def __init__(self, lambda1, lambda2):
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def penalty(self, w):
        return self.lambda1 * np.sum(np.abs(w)) + self.lambda2 * np.sum(w**2)

    def quadratic_approximation(self, w, eps=1e-8):
        diag_entries = 1.0 / (np.abs(w) + eps)
        return self.lambda1 * np.diag(diag_entries) + self.lambda2 * np.eye(len(w))
    
    def __repr__(self):
        return f"{self.__class__.__name__}(lambda1={self.lambda1}, lambda2={self.lambda2})"


# 4. Group LASSO norm class

class GroupLassoNorm(BaseNorm):
    """
    Implements the Group LASSO norm:
         Omega(w) = sum_{g} weight_g * ||w[g]||₂,
    where 'groups' is a list of lists (each inner list contains indices for a group).
    The optional 'weights' parameter allows using group-specific weights.
    """
    def __init__(self, groups, weights=None):
        self.groups = groups  # For example, groups = [[0,1,2], [3,4,5], ...]
        if weights is None:
            weights = [np.sqrt(len(g)) for g in groups]
        self.weights = weights

    def penalty(self, w):
        total = 0.0
        for g, weight in zip(self.groups, self.weights):
            total += weight * np.linalg.norm(w[g])
        return total

    def quadratic_approximation(self, w, eps=1e-8):
        p = len(w)
        Q = np.zeros((p, p))
        for g, weight in zip(self.groups, self.weights):
            w_group = w[g]
            norm_group = np.linalg.norm(w_group)
            if norm_group < eps:
                norm_group = eps
            factor = weight / (norm_group + eps)
            for idx in g:
                Q[idx, idx] = factor
        return Q
    
    def __repr__(self):
        return f"{self.__class__.__name__}(groups={self.groups}, weights={self.weights})"



# 5. Generic MM dual update routine

def mm_dual_norm_generic(v, norm, rho=10.0, tol=1e-6, max_iter=100, eps=1e-8, ppnu=0.8):
    """
    Compute the dual update with a generic norm via an MM algorithm.
    
    We wish to solve:
          max_{w}  <v, w>  subject to  norm.penalty(w) <= 1,
    or equivalently, minimize:
          L(w) = - v^T w - rho * log(1 - norm.penalty(w)).
    
    The algorithm also applies a sparsity thresholding to v controlled by ppnu.
    
    Parameters:
      v       : numpy.ndarray, the input vector.
      norm    : an instance of BaseNorm (or subclass).
      rho     : barrier parameter.
      tol     : convergence tolerance.
      max_iter: maximum MM iterations.
      eps     : small constant for numerical stability.
      ppnu    : proportion of coordinates to shrink (between 0 and 1).
      
    Returns:
      w       : optimized weight vector.
      dual_val: the dual value (v^T w).
    """
    p = v.shape[0]

    # --- Step 1: Thresholding to promote sparsity ---
    abs_v = np.abs(v)
    sorted_abs = np.sort(abs_v)
    index_thresh = int(np.floor(ppnu * p))
    index_thresh = min(max(index_thresh, 0), p - 1)
    nu = sorted_abs[index_thresh]
    v_thresh = np.sign(v) * np.maximum(np.abs(v) - nu, 0)
    v = v_thresh

    # --- Step 2: MM algorithm initialization ---
    w = np.ones(p) * 0.1  # small nonzero initialization
    for _ in range(max_iter):
        A = norm.penalty(w)
        if A >= 1:
            w *= 0.99 / (A + eps)
            A = norm.penalty(w)
        upsilon = 1 - A  # feasibility gap

        # Quadratic surrogate from the provided norm.
        Q = norm.quadratic_approximation(w, eps)
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



# 6. Dual-sPLS with generic norm (d_spls_generic)

def d_spls_generic(X, y, ncp, norm, rho=10.0, ppnu=0.8, verbose=True):
    """
    Compute a Dual-sPLS model using an MM algorithm with a generic norm.
    
    Parameters:
      X      : numpy.ndarray of shape (n_samples, n_features)
      y      : numpy.ndarray of shape (n_samples,)
      ncp    : int, number of latent components.
      norm   : an instance of BaseNorm (or subclass) defining the penalty.
      rho    : barrier parameter for MM updates.
      ppnu   : proportion of coordinates to shrink for sparsity.
      verbose: bool, if True prints iteration details.
      
    Returns:
      A dictionary with the following keys:
        'Xmean'         : mean of X columns.
        'scores'        : latent scores.
        'loadings'      : weight loadings.
        'Bhat'          : regression coefficients.
        'intercept'     : intercept values.
        'fitted_values' : model fitted values.
        'residuals'     : residuals.
        'dualvals'      : computed dual norm values for each component.
        'zerovar'       : count of zero coefficients per component.
        'ind_diff0'     : indices of nonzero coefficients per component.
        'type'          : string indicating model type.
    """
    # Center data.
    X = np.array(X)
    y = np.ravel(y)
    n, p = X.shape
    Xm = np.mean(X, axis=0)
    Xc = X - Xm
    ym = np.mean(y)
    yc = y - ym

    # Initialize model storage.
    WW = np.zeros((p, ncp))
    TT = np.zeros((n, ncp))
    Bhat = np.zeros((p, ncp))
    YY = np.zeros((n, ncp))
    RES = np.zeros((n, ncp))
    intercept = np.zeros(ncp)
    zerovar = np.zeros(ncp, dtype=int)
    dualvals = np.zeros(ncp)
    ind_diff0 = [None] * ncp

    # Deflated copy of X.
    Xdef = Xc.copy()

    for ic in range(ncp):
        # Compute auxiliary vector from deflated design.
        z = Xdef.T @ yc

        # Use the generic MM algorithm for the dual update.
        w, dualval = mm_dual_norm_generic(z, norm, rho=rho, tol=1e-6, max_iter=100,
                                            eps=1e-8, ppnu=ppnu)
        WW[:, ic] = w
        dualvals[ic] = dualval

        # Compute score vector and normalize.
        t_vec = Xdef @ w
        norm_t = np.linalg.norm(t_vec, 2)
        t_vec_norm = t_vec if norm_t == 0 else t_vec / norm_t
        TT[:, ic] = t_vec_norm

        # Deflation step to update Xdef.
        Xdef = Xdef - np.outer(t_vec_norm, t_vec_norm) @ Xdef

        # Update regression coefficients from latent scores.
        TT_current = TT[:, :ic+1]
        WW_current = WW[:, :ic+1]
        R = (TT_current.T @ Xc) @ WW_current
        # Force R to be upper-triangular.
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
            print(f"[MM Dual-sPLS] Component {ic+1}, Dual norm = {dualval:.4f}")

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
        'type': 'generic'
    }


