import numpy as np
import matplotlib.pyplot as plt

def d_spls_plot(mod_dspls, ncomp):
    """
    Plots the coefficient curve of a Dual-SPLS regression.
    
    Parameters:
        mod_dspls: dict
            A fitted Dual-SPLS object. It should contain 'Xmean', 'zerovar', and 'Bhat' as keys.
        ncomp: list or int
            A list or a single integer representing the number of Dual-SPLS components to consider.
    
    Returns:
        None
    """
    Xmean = mod_dspls['Xmean']
    p = len(Xmean)

    # Set up the plot layout (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Plot the mean of the original data
    axes[0].plot(Xmean, label="Mean of the original data", color='b')
    axes[0].set_title("Mean of the original data")
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('X')

    # Dual-SPLS plot for each component in ncomp
    for i in range(len(ncomp)):
        comp_idx = ncomp[i]
        nz = mod_dspls['zerovar'][comp_idx]
        
        # Plot the Dual-SPLS coefficients
        axes[1].plot(range(1, p+1), mod_dspls['Bhat'][:, comp_idx], label=f"Dual-SPLS (ncp = {comp_idx})", color='g')
        non_zero_indices = np.where(mod_dspls['Bhat'][:, comp_idx] != 0)[0]
        axes[1].scatter(non_zero_indices + 1, mod_dspls['Bhat'][non_zero_indices, comp_idx], color='red', label="Non-zero values")
    
    axes[1].set_title(f"Dual-SPLS Coefficients")
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Coefficients')

    # Add legend
    axes[1].legend(loc="upper right", fontsize=8)
    axes[0].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == '__main__':
    # mod_dspls is assumed to be a dictionary containing 'Xmean', 'zerovar', and 'Bhat'
    mod_dspls = {
        'Xmean': np.random.rand(100),  # Example mean of the original data
        'zerovar': np.random.randint(0, 10, 10),  # Example zero variance counts
        'Bhat': np.random.rand(100, 10)  # Example Dual-SPLS coefficients
    }

    # Call the function with a list of components to plot
    d_spls_plot(mod_dspls, [5, 6, 7])


