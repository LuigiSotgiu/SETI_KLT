import numpy as np
import matplotlib.pyplot as plt

def plot_sync_phase_viz(is_synchronous=True):
    # Parameters
    W = 100  # Window length
    K = 6    # Number of observation windows (realizations)
    
    if is_synchronous:
        # Exact integer number of cycles per window
        f = 2.0 / W 
        title = r"Synchronous Regime ($f_0 \cdot W \in \mathbb{Z}$)"
        cmap_color = "RdBu"
    else:
        # Fractional number of cycles per window
        f = 2.3 / W
        title = r"Asynchronous Regime ($f_0 \cdot W \notin \mathbb{Z}$)"
        cmap_color = "RdBu"

    # Signal Generation
    t = np.arange(K * W)
    sig = np.sin(2 * np.pi * f * t)
    matrix = sig.reshape(K, W)
    
    # Statistical Centering
    col_mean = np.mean(matrix, axis=0)
    centered_matrix = matrix - col_mean

    # Plotting Setup
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(title, fontsize=22, fontweight='bold', y=0.95)

    # 1. Trajectory Matrix Visualization
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    im1 = ax1.imshow(matrix, aspect='auto', cmap=cmap_color)
    ax1.set_title(r"1. Trajectory Matrix $X$ (Windowed Observations)", fontsize=14, fontweight='bold')
    ax1.set_ylabel(r"Realization Index ($K$)")
    plt.colorbar(im1, ax=ax1)

    # 2. Column-wise Mean Vector
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    ax2.plot(col_mean, color='navy', lw=3)
    ax2.set_title(r"2. Column-wise Mean $\mu_X$ (Statistical Bias)", fontsize=14, fontweight='bold')
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.3)

    # 3. Centered Matrix Result
    ax3 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    im3 = ax3.imshow(centered_matrix, aspect='auto', cmap=cmap_color)
    ax3.set_title(r"3. Centered Matrix $(X - \mu_X)$", fontsize=14, fontweight='bold')
    ax3.set_ylabel(r"Realization Index ($K$)")
    ax3.set_xlabel(r"Time Samples ($W$)")
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# Generate both figures
fig_sync = plot_sync_phase_viz(is_synchronous=True)
fig_async = plot_sync_phase_viz(is_synchronous=False)

plt.show()