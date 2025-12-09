import numpy as np
from scipy.signal import hilbert

def fourier_phase_randomized(x:np.ndarray, N:int=1, seed:int=None)->np.ndarray:
    """
    Generate N surrogates of an input signal on the time domain.

    Parameters
    ----------
    x : np.ndarray [n_samples]
        Time series of the original signal.
    N : int
        Number of surrogates to generate.
    seed : int
        Optional seed for the random generator.

    Returns
    -------
    y : np.ndarray[n_samples, N]
        Surrogates of the input signal.
    """

    rng = np.random.default_rng(seed)

    n = len(x)
    half = n // 2
    X_amp = np.abs(np.fft.fft(x))

    Yf = np.zeros((n, N), dtype=complex)
    Yf[0] = X_amp[0] # DC component

    phases = rng.uniform(0, 2*np.pi, size=(half - 1, N))
    Yf[1:half] = X_amp[1:half, None] * np.exp(1j * phases)

    if n % 2 == 0:
        Yf[half] = X_amp[half] # Nyquist frequency
        Yf[half+1:] = np.conj(Yf[1:half][::-1])
    else:
        Yf[half+1:] = np.conj(Yf[1:half+1][::-1])

    y = np.real(np.fft.ifft(Yf, axis=0))
    
    return y


def IAAFT(x:np.ndarray, N:int=1, n_iter:int=10, seed:int=None)->np.ndarray:
    """
    Generate N IAAFT surrogates for signal x.
    
    Parameters
    ----------
    x : np.ndarray [n_samples]
        Time series of the original signal.
    N : int
        Number of surrogates.
    n_iter : int
        Number of IAAFT iterations.
    seed : int
        Optional seed for the random generator.
    
    Returns
    -------
    y : np.ndarray [n_samples, N]
        IAAFT surrogate signals.
    """

    x_sorted = np.sort(x)
    X_amp = np.abs(np.fft.fft(x))

    y = fourier_phase_randomized(x, N, seed)

    for _ in range(n_iter):
        # Impose the exact time-domain amplitude distribution
        idx = np.argsort(y, axis=0)
        for k in range(N):
            y[idx[:, k], k] = x_sorted # sort y and replace with sorted original x

        # Impose Fourier amplitudes but keep phases
        Yf = np.fft.fft(y, axis=0)
        phases = Yf / (np.abs(Yf) + 1e-15)
        Yf = X_amp[:, None] * phases # impose original amplitude spectrum
        y = np.real(np.fft.ifft(Yf, axis=0))

    return y


def bivariate_IAAFT(x: np.ndarray,
                    y: np.ndarray,
                    N: int = 1,
                    n_iter: int = 10,
                    seed: int = None):
    """
    Generate N bivariate IAAFT surrogate pairs for signals x and y.

    Parameters
    ----------
    x, y : np.ndarray [n_samples]
        Original signals.
    N : int
        Number of surrogate pairs.
    n_iter : int
        Number of IAAFT iterations.
    seed : int
        Optional random seed.

    Returns
    -------
    xs, ys : np.ndarray [n_samples, N]
        Bivariate IAAFT surrogates of x and y.
    """

    rng = np.random.default_rng(seed)

    n = len(x)

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    # Target Fourier amplitudes (univariate)
    X_amp = np.abs(np.fft.fft(x))
    Y_amp = np.abs(np.fft.fft(y))

    # Initial joint random permutation (preserves cross-correlation structure)
    perm = rng.permutation(n)

    xs = np.zeros((n, N))
    ys = np.zeros((n, N))

    for k in range(N):
        xs[:, k] = x_sorted[perm]
        ys[:, k] = y_sorted[perm]

        # Apply random Fourier phases
        Xf = np.fft.fft(xs[:, k])
        Yf = np.fft.fft(ys[:, k])
        rand_phases = np.exp(1j * rng.uniform(0, 2*np.pi, n))
        xs[:, k] = np.real(np.fft.ifft(np.abs(Xf) * rand_phases))
        ys[:, k] = np.real(np.fft.ifft(np.abs(Yf) * rand_phases))

    for _ in range(n_iter):

        # Impose exact amplitude distribution for each surrogate
        for k in range(N):
            idx_x = np.argsort(xs[:, k])
            idx_y = np.argsort(ys[:, k])

            xs[idx_x, k] = x_sorted
            ys[idx_y, k] = y_sorted

        # Impose Fourier amplitudes but keep phases
        Xsf = np.fft.fft(xs, axis=0)  # shape [n_samples, N]
        Ysf = np.fft.fft(ys, axis=0)

        X_phase = Xsf / (np.abs(Xsf) + 1e-15)
        Y_phase = Ysf / (np.abs(Ysf) + 1e-15)

        X_new = X_amp[:, None] * X_phase
        Y_new = Y_amp[:, None] * Y_phase

        xs = np.real(np.fft.ifft(X_new, axis=0))
        ys = np.real(np.fft.ifft(Y_new, axis=0))

    return xs, ys


def randomize_phase_velocity(phase:np.ndarray, N:int=1, seed:int=None, return_idx:bool=False)->np.ndarray:
    """
    Creates surrogates of a phase array by permuting the phase velocities.
    
    Parameters
    ----------
    phase : np.ndarray [n_samples]
        Hilbert phase.
    N : int
        Number of surrogates to generate.
    seed : int
        Optional seed for the random generator.

    Returns
    -------
    randomized_phase : np.ndarray[n_samples, N]
        Surrogates of the phase.
    """

    rng = np.random.default_rng(seed)

    idx = np.column_stack([rng.permutation(len(phase) - 1) for _ in range(N)])

    velocity = np.diff(phase)
    velocity_permut = velocity[idx]
    randomized_phase = np.append(phase[0], phase[0] + np.cumsum(velocity_permut))

    if return_idx:
        return randomized_phase, idx
    else:
        return randomized_phase


def hilbert_velocity_randomized(x:np.ndarray, N:int=1, seed:int=None, return_hilbert:bool=False)->np.ndarray:
    """
    Generate N surrogates of an input signal on the Hilbert phase domain.

    Parameters
    ----------
    x : np.ndarray [n_samples]
        Time series of the original signal.
    N : int
        Number of surrogates to generate.
    seed : int
        Optional seed for the random generator.

    Returns
    -------
    y : np.ndarray[n_samples, N]
        Surrogates of the input signal.
    """

    analytical = hilbert(x)
    amplitude = np.abs(analytical)
    phase = np.unwrap(np.angle(analytical))

    randomized_phase = randomize_phase_velocity(phase, N, seed)
    
    y = amplitude*np.cos(randomized_phase)

    if return_hilbert:
        return y, phase, randomized_phase, amplitude
    else:
        return y