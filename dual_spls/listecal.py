import numpy as np
def d_spls_listecal(Datatype, pcal):
    """
    Computes, for each group in Datatype, the number of observations to include in the calibration set.
    If pcal is given as a percentage, it selects floor(pcal/100 * n_group).
    
    Parameters
    ----------
    Datatype : array-like
        Group label for each observation.
    pcal : float
        Calibration percentage (0-100).
    
    Returns
    -------
    list
        A list of counts for each group.
    """
    Datatype = np.array(Datatype)
    unique_types = np.unique(Datatype)
    Listecal = []
    for t in unique_types:
        n_group = np.sum(Datatype == t)
        n_select = int(np.floor(pcal/100. * n_group))
        Listecal.append(n_select)
    return Listecal
