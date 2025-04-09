import numpy as np

def d_spls_type(y, ncells=10):
    """
    Assigns a type (group label) to each observation based on the sorted response values.
    
    Parameters
    ----------
    y : array-like
        Response vector.
    ncells : int, default=10
        Number of groups to split the data into.
    
    Returns
    -------
    np.ndarray
        An array of group labels (1-indexed, as in the original R code).
    """
    y = np.array(y).flatten()
    n = len(y)
    sorted_idx = np.argsort(y)
    # Divide into ncells groups (approximately equal in number)
    group_size = int(np.floor(n / ncells))
    Datatype = np.zeros(n, dtype=int)
    for i in range(ncells):
        if i == ncells - 1:
            Datatype[sorted_idx[i*group_size:]] = i + 1
        else:
            Datatype[sorted_idx[i*group_size:(i+1)*group_size]] = i + 1
    return Datatype
