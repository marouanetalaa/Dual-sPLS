def d_spls_print(model):
    """
    Prints a summary of a dual-SPLS model.
    
    Parameters
    ----------
    model : dict
        A dual-SPLS model dictionary.
    """
    print("Dual-SPLS Model Summary:")
    print("------------------------")
    print(f"Type: {model.get('type', 'unknown')}")
    print("Coefficients per component:")
    for comp, intercept in enumerate(model['intercept'], start=1):
        print(f" Component {comp}: intercept = {intercept:.4f}, nonzero loadings: {(model['Bhat'][:, comp-1]!=0).sum()}")
