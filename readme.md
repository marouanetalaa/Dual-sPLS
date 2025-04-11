# How to Implement a New Penalty Norm Using BaseNorm

## Create a New Class

- **Inherit from the BaseNorm abstract class.**

## Implement the `penalty(w)` Method

- **Define the mathematical expression of your norm.**
- This method computes $$\Omega(w)$$ given a weight vector w.

## Implement the `quadratic_approximation(w, eps)` Method

- **Develop a smooth surrogate for your norm,** often a diagonal or block-diagonal matrix $$Q$$ such that

  $$
  \Omega(w) \approx w^T Q w.
  $$

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

