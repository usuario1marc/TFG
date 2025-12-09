import numpy as np


def simulate(theta:float=1.0, mu:float=0.0, sigma:float=1.0, N:int=1, n_samples:int=1000, start:int=100, dt:float=0.1, seed:int=None)->np.ndarray:
    """
    Generate N Ornstein-Uhlenbeck (OU) processes.

    Parameters
    ----------
    theta : float
        Mean reversion rate, >0
    mu : float
        Long-term mean.
    sigma : float
        Diffusion coefficient of the OU process.
    N : int
        Number of OU processes to generate.
    n_samples : int
        Number of samples to return (after burn-in).
    start : int
        Number of initial samples to discard (to allow stabilization).
    dt : float
        Time step represented by each sample.
    seed : int
        Optional random seed.

    Returns
    -------
    np.ndarray [n_samples x 1]
        Time vector.
    np.ndarray [n_samples x N]
        Generated OU processes.
    """

    if seed is not None:
        np.random.seed(seed)

    total = n_samples + start
    x = np.zeros((total, N))

    exp_term = np.exp(-theta * dt)
    var_factor = (sigma**2) * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
    std_term = np.sqrt(var_factor)
    x[0, :] = mu

    for t in range(1, total):
        noise = std_term * np.random.randn(N)
        x[t, :] = mu + (x[t-1, :] - mu) * exp_term + noise

    return np.arange(n_samples) * dt, x[start:, :]


if __name__ == "__main__":
    # Example: theta=1.2, mu=0.0
    theta = 1.2
    mu = 0.0
    sigma = 1.0
    t, X = simulate(theta=theta, mu=mu, sigma=sigma, N=2)

    np.savez("simulations/ou.npz", X=X, t=t, args={"theta": theta, "mu": mu, "sigma": sigma})
    print(f"OU process saved as 'ou.npz' with shape {X.shape}")
