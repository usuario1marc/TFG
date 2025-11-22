import numpy as np


def simulate(phi:list|np.ndarray, sigma:float=1.0, N:int=1, n_samples:int=1000, start:int=100, dt:float=0.1, seed:int=None)->np.ndarray:
    """
    Generate N autoregressive (AR) processes of order P.
    
    Parameters
    ----------
    phi : list or np.ndarray [P x 1]
        Coefficients of the AR process.
    sigma : float
        Standard deviation of the white noise.
    N : int
        Number of AR processes to generate.
    n_samples : int
        Number of samples to return (after burn-in).
    start : int
        Number of initial samples to discard (to allow stabilization).
    dt : float
        Time step represented by each sample.
    seed : int
        Optionally, you may set a seed for the noise.
        
    Returns
    -------
    np.ndarray [n_samples x 1]
        Time vector.
    np.ndarray [n_samples x N]
        Generated AR.
    """

    if seed is not None:
        np.random.seed(seed)

    p = len(phi)
    eps = np.random.normal(scale=sigma, size=(n_samples + start, N))
    x = np.zeros((n_samples + start, N))

    for t in range(p, n_samples + start):
        x[t, :] = np.dot(phi, x[t-p:t, :][::-1, :]) + eps[t, :]

    return np.arange(n_samples)*dt, x[start:, :]


if __name__ == "__main__":
    phi = [0.9]
    sigma = 1.0
    t, X = simulate(phi=phi, sigma=sigma, N=2)

    np.savez("simulations/ar.npz", X=X, t=t, args={"phi":phi, "sigma":sigma})
    print(f"AR process saved as 'ar.npz' with shape {X.shape}")