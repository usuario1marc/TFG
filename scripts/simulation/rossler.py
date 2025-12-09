import numpy as np
from scipy.integrate import solve_ivp   #https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html


def rossler(t, state, wx=1, wy=1, eyx=0, exy=0):
    """
    Computes the derivative of the Rössler dynamics given a state.
    """
    x1, x2, x3, y1, y2, y3 = state

    dx1dt = -wx*x2 - x3 + eyx*(y1-x1)
    dx2dt = wx*x1 + 0.165*x2
    dx3dt = 0.2 + x3 * (x1 - 10)

    dy1dt = -wy*y2 - y3 + exy*(x1-y1)
    dy2dt = wy*y1 + 0.165*y2
    dy3dt = 0.2 + y3 * (y1 - 10)

    return [dx1dt, dx2dt, dx3dt, dy1dt, dy2dt, dy3dt]


def euler(f, t_span:tuple, state_0:np.ndarray, dt:float, args:tuple=None, sigma:float=0)->np.ndarray:
    """
    Integrates a dynamic using Stochastic Euler method.
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    state_vec = np.zeros((t_eval.size, state_0.size))
    state_vec[0, :] = state_0

    for i, t in enumerate(t_eval):
        if i==0:
            continue
        gradient = f(t, state_vec[i-1, :], *args)
        noise = np.random.random(len(state_0)) * np.array([1, 1, 0, 1, 1, 0]) # Noise in x1, x2, y1, y2
        state_vec[i, :] = state_vec[i-1, :] + np.array(gradient) * dt + noise * np.sqrt(dt) * sigma

    return state_vec


def simulate(wx:float=0.5, wy:float=0.5, eyx:float=0, exy:float=0, sigma:float=0, n_samples:int=1000, start:int=100, downsample:int=100, dt:int=0.001, seed:int=None):
    """
    Generate two noisy coupled Rössler dynamics (X and Y).
    
    Parameters
    ----------
    wx : float
        Natural angular frequency of dynamic X.
    wy : float
        Natural angular frequency of dynamic Y.
    eyx : float
        Coupling strength onto dynamic X by dynamic Y.
    exy : float
        Coupling strength onto dynamic Y by dynamic X.
    sigma : float
        Standard deviation of the white noise.
    n_samples : int
        Number of samples to return (after downsample and burn-in).
    start : int
        Number of initial samples to discard (to allow stabilization).
    dt : float
        Time step represented by each sample.
    seed : int
        Optionally, you may set a seed.
    
    Returns
    -------
    np.ndarray [n_samples, 6]
        Dynamics of the X system (x1, x2, x3) and the Y system (y1, y2, y3).
    """

    if seed is not None:
        np.random.seed(seed)

    t_span = (0, (n_samples + start) * dt * downsample)
    state_0 = np.random.rand(6)
    sol = euler(rossler, t_span=t_span, state_0=state_0, dt=dt, args=(wx, wy, eyx, exy), sigma=sigma)

    t = np.arange(n_samples) * (dt * downsample)
    sol_downsampled = sol[::downsample][-n_samples:]
    return t, sol_downsampled


def simulate_multi(N:int=10, wx:float=0.5, wy:float=0.5, eyx:float=0, exy:float=0, sigma:float=0, n_samples:int=1000, start:int=100, downsample:int=100, dt:int=0.001, seed:int=None):
    """
    Generate N pairs of noisy coupled Rössler dynamics (X and Y).
    
    Parameters
    ----------
    N : int
        Number of pairs of signals to simulate.
    wx : float
        Natural angular frequency of dynamic X.
    wy : float
        Natural angular frequency of dynamic Y.
    eyx : float
        Coupling strength onto dynamic X by dynamic Y.
    exy : float
        Coupling strength onto dynamic Y by dynamic X.
    sigma : float
        Standard deviation of the white noise.
    n_samples : int
        Number of samples to return (after downsample and burn-in).
    start : int
        Number of initial samples to discard (to allow stabilization).
    dt : float
        Time step represented by each sample.
    seed : int
        Optionally, you may set a seed.
    
    Returns
    -------
    np.ndarray [n_samples, 6, N]
        Dynamics of the X system (x1, x2, x3) and the Y system (y1, y2, y3).
    """

    if seed is not None:
        np.random.seed(seed)

    sol = np.zeros((n_samples, 6, N))
    for i in range(N):
        t, sol[:, :, i] = simulate(wx=wx, wy=wy, eyx=eyx, exy=exy, sigma=sigma, n_samples=n_samples, start=start, downsample=downsample, dt=dt)

    return t, sol


if __name__ == "__main__":
    wx = 0.5
    wy = 0.5
    eyx = 0
    exy = 0
    t, X = simulate(wx=wx, wy=wy, eyx=eyx, exy=exy)

    np.savez("simulations/rossler.npz", X=X, t=t, args={"wx":wx, "wy":wy, "eyx":eyx, "exy":exy})
    print(f"Coupled Rossler dynamics saved as 'rossler.npz' with shape {X.shape}")