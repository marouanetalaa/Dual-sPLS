# How to Implement a New Penalty Norm Using BaseNorm

## Create a New Class

- **Inherit from the BaseNorm abstract class.**

## Implement the `penalty(w)` Method

- **Define the mathematical expression of your norm.**
- This method computes $$\Omega(w)$$ given a weight vector w.

## Implement the `quadratic_approximation(w, eps)` Method

- **Develop a smooth surrogate for your norm,** often a diagonal or block-diagonal matrix $$Q$$ such that

$$\Omega(w) \approx w^T Q w.$$

- Use a small constant `eps` for numerical stability.

## (Optional) Override `__repr__`

- Provide a custom string representation for clearer debugging and logging.


## Example (Lasso)

  ```python
    class LassoNorm(BaseNorm):
        
        #Implements the LASSO (ℓ₁) norm:
        #     Omega(w) = lambda1 * ||w||₁.
        #The quadratic surrogate for each coordinate is given by
        #     |w_i| ≈ w_i^2/(|w_i| + eps)
        #to smooth the non-differentiability.
        
        def __init__(self, lambda1):
            self.lambda1 = lambda1

        def penalty(self, w):
            return self.lambda1 * np.sum(np.abs(w))

        def quadratic_approximation(self, w, eps=1e-8):
            diag_entries = 1.0 / (np.abs(w) + eps)
            return self.lambda1 * np.diag(diag_entries)
        
        def __repr__(self):
            return f"{self.__class__.__name__}(lambda1={self.lambda1})"

```


## Example Usage


```python
import numpy as np

from dual_spls.plug_norm import ElasticNetNorm, d_spls_generic, mm_dual_norm_generic


if __name__ == '__main__':
    np.random.seed(0)
    n_samples, n_features = 100, 20
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    ncp = 3  # number of latent components

    # --- Example with Elastic Net norm ---
    enet_norm = ElasticNetNorm(lambda1=0.5, lambda2=0.1)
    w_enet, dual_val_enet = mm_dual_norm_generic(np.random.randn(n_features), enet_norm, rho=10.0, ppnu=0.8)
    print("Elastic Net dual update:")
    print("w =", w_enet)
    print("Dual value =", dual_val_enet)

    # Run the dual-sPLS procedure with the Elastic Net norm.
    model = d_spls_generic(X, y, ncp, enet_norm, rho=10.0, ppnu=0.8, verbose=True)
    print("\nDual-sPLS model output keys:", model.keys())
```

## References

1. **M. Bernardi, M. Stefanucci, and A. Canale.**  
   *Numerical evaluation of dual norms via the MM algorithm.* Preprint, 2020.

2. **L. Alsouki, L. Duval, C. Marteau, R. El Haddad, and F. Wahl.**  
   *Dual-sPLS: a family of Dual Sparse Partial Least Squares regressions for feature selection and prediction with tunable sparsity; evaluation on simulated and near-infrared (NIR) data.* Preprint, 2023.



