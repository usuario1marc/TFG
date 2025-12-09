import numpy as np
from scipy.signal import hilbert

def measures(phase:np.ndarray=None, signal:np.ndarray=None, dt:float=1.0) -> tuple[np.ndarray]:
    """
    Compute univariate phase-based measures as defined in:
    Espinoso & Andrzejak (2022), Phase Irregularity.

    Parameters
    ----------
    phase : np.ndarray [N]
        Instantaneous phase φ(t). MUST BE UNWRAPPED.
    dt : float
        Sampling interval Δt. Default = 1.0.

    Returns
    -------
    V : float
        Coefficient of phase velocity variation  (S / M)
    M : float
        Mean phase velocity
    S : float
        Standard deviation of phase velocity
    Ω : np.ndarray [N-1]
        Instantaneous phase velocity time series
    """

    if phase is None and signal is None:
        raise ValueError("Either an input signal or an unwrapped phase array must be provided.")
    elif phase is None:
        phase = np.unwrap(np.angle(hilbert(signal)))

    # Instantaneous phase velocity Ω(t_j) = (Φ_{j+1} - Φ_j) / Δt
    dphi = np.diff(phase)
    omega = dphi / dt

    # Mean phase velocity M
    M = np.mean(omega)

    # Standard deviation S
    S = np.std(omega)

    # Coefficient of phase velocity variation V = S / M
    # (Edge case: if M = 0, set V = np.inf)
    V = S / M if M != 0 else np.inf

    return V, M, S, omega


def MPC(phase_x:np.ndarray, phase_y:np.ndarray)->float:
    """
    Compute the Mean Phase Coherence (bivariate phase measure) between two signals.

    Parameters
    ----------
    phase_x : np.ndarray [N]
        Instantaneous phase ϕ_x(t) of signal x. Can be wrapped or unwrapped.
    phase_y : np.ndarray [N]
        Instantaneous phase ϕ_y(t) of signal y. Can be wrapped or unwrapped.

    Returns
    -------
    R : float
        Mean phase coherence (0 ≤ R ≤ 1)
    """

    if len(phase_x) != len(phase_y):
        raise ValueError("phase_x and phase_y must have the same length.")

    # Relative phase difference φ(t)
    phi = phase_x - phase_y

    # Mean phase coherence:
    # R = | (1/N) * Σ exp(i * φ(t)) |
    R = np.abs(np.mean(np.exp(1j * phi)))

    return R


def reconstruct(amplitude:np.ndarray, phase:np.ndarray)->tuple[np.ndarray]:
    """
    Reconstruct original signal and Hilbert transform from the Analytic Signal.
    
    Parameters
    ----------
    amplitude : np.ndarray [N x 1]
        Instantaneous amplitude of the Analytic Signal.
    phase : np.ndarray [N x 1]
        Instantaneous phase of the Analytic Signal.
        
    Returns
    -------
    real : np.ndarray [N x 1]
        Original signal in the time domain.
    imag: np.ndarray [N x 1]
        Hilbert transform of the signal.
    """

    real = np.cos(phase) * amplitude # Original signal
    imag = np.sin(phase) * amplitude # Hilbert transform

    return real, imag