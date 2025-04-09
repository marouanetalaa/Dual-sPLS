import numpy as np
def norm1(vec):
    return np.sum(np.abs(vec))

def norm2(vec):
    return np.sqrt(np.sum(vec**2))
