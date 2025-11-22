import numpy as np

def generate_ar_process(phi, n_samples=1000, sigma=1.0, burn_in=100):
    """
    Generate an autoregressive (AR) process of order p.
    
    Parameters
    ----------
    phi : list or np.ndarray
        Coefficients of the AR process (e.g., [0.9] for AR(1), [0.75, -0.25] for AR(2)).
    n_samples : int
        Number of samples to return (after burn-in).
    sigma : float
        Standard deviation of the white noise.
    burn_in : int
        Number of initial samples to discard (to allow stabilization).
    
    Returns
    -------
    np.ndarray
        The generated AR time series.
    """
    p = len(phi)
    eps = np.random.normal(scale=sigma, size=n_samples + burn_in)
    x = np.zeros(n_samples + burn_in)

    # Generate the AR process
    for t in range(p, n_samples + burn_in):
        x[t] = np.dot(phi, x[t-p:t][::-1]) + eps[t]

    return x[burn_in:]  # discard burn-in period


if __name__ == "__main__":
    phi = [0.9]
    n_samples = 1000
    sigma = 1.0

    ar_series = generate_ar_process(phi, n_samples, sigma)

    # Save as NumPy array
    np.savez("ar_process.npz", x=ar_series, t=np.arange(0, len(ar_series)), args=(0.9,))
    print(f"AR process saved as 'ar_process.npy' with shape {ar_series.shape}")
