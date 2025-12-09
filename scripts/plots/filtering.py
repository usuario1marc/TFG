import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scripts.utils import phase


def bw_sweep(
    x_signal: np.ndarray,
    y_signal: np.ndarray,
    bw_array: np.ndarray = None,
    fc: float = 10,
    fs: float = 100
):
    """
    Computes MPC(bw_x, bw_y).

    Parameters
    ----------
    x_signal : np.ndarray [n_samples, n_signals]
        Array of signals.
    y_signal : np.ndarray [n_samples, n_signals]
        Array of signals.
    bw_array : np.ndarray [n_bandwidths]
        Bandwidth.
    fc : float
        Central frequency.
    fs : float
        Sampling frequency.

    Returns
    -------
    MPC_matrix : np.ndarray [len(bw_array), len(bw_array)]
    bw_array : np.ndarray
    """

    if bw_array is None:
        bw_array = np.arange(0.1, 6, 0.1)

    try:
        n_signals = x_signal.shape[1]
    except IndexError:
        n_signals=1
    n_bw = bw_array.shape[0]
    MPC_matrix = np.zeros((n_bw, n_bw))

    # Pre-store phases
    x_phases = []
    for bw in bw_array:
        sos = sg.butter(4, [fc - bw/2, fc + bw/2], btype='bandpass', fs=fs, output='sos')
        x_filt = sg.sosfiltfilt(sos, x_signal, axis=0)
        x_phase = np.unwrap(np.angle(sg.hilbert(x_filt, axis=0)), axis=0)
        x_phases.append(x_phase)
    y_phases = []
    for bw in bw_array:
        sos = sg.butter(4, [fc - bw/2, fc + bw/2], btype='bandpass', fs=fs, output='sos')
        y_filt = sg.sosfiltfilt(sos, y_signal, axis=0)
        y_phase = np.unwrap(np.angle(sg.hilbert(y_filt, axis=0)), axis=0)
        y_phases.append(y_phase)

    # Compute MPC
    for i in range(n_bw):
        for j in range(n_bw):
            x_phase = x_phases[i]
            y_phase = y_phases[j]
            mpc_values = []
            for k in range(n_signals):
                mpc = phase.MPC(x_phase[:,k], y_phase[:,k])
                mpc_values.append(mpc)

            MPC_matrix[i, j] = np.mean(mpc_values)

    return MPC_matrix, bw_array


def fc_sweep(
    x_signal: np.ndarray,
    y_signal: np.ndarray,
    fc_array: np.ndarray = None,
    bw: float = 4,
    fs: float = 100
):
    """
    Computes MPC(bw_x, bw_y).

    Parameters
    ----------
    x_signal : np.ndarray [n_samples, n_signals]
        Array of signals.
    y_signal : np.ndarray [n_samples, n_signals]
        Array of signals.
    fc_array : np.ndarray [n_bandwidths]
        Central frequency.
    bw : float
        Bandwidth.
    fs : float
        Sampling frequency.

    Returns
    -------
    MPC_matrix : np.ndarray [len(fc_array), len(fc_array)]
    fc_array : np.ndarray
    """

    if fc_array is None:
        fc_array = np.arange(4, 10, 0.1)

    try:
        n_signals = x_signal.shape[1]
    except IndexError:
        n_signals=1
    n_fc = fc_array.shape[0]
    MPC_matrix = np.zeros((n_fc, n_fc))

    # Pre-store phases
    x_phases = []
    for fc in fc_array:
        sos = sg.butter(4, [fc - bw/2, fc + bw/2], btype='bandpass', fs=fs, output='sos')
        x_filt = sg.sosfiltfilt(sos, x_signal, axis=0)
        x_phase = np.unwrap(np.angle(sg.hilbert(x_filt, axis=0)), axis=0)
        x_phases.append(x_phase)
    y_phases = []
    for fc in fc_array:
        sos = sg.butter(4, [fc - bw/2, fc + bw/2], btype='bandpass', fs=fs, output='sos')
        y_filt = sg.sosfiltfilt(sos, y_signal, axis=0)
        y_phase = np.unwrap(np.angle(sg.hilbert(y_filt, axis=0)), axis=0)
        y_phases.append(y_phase)

    # Compute MPC
    for i in range(n_fc):
        for j in range(n_fc):
            x_phase = x_phases[i]
            y_phase = y_phases[j]

            if n_signals == 1:
                MPC_matrix[i, j] = phase.MPC(x_phase, y_phase)
            else:
                mpc_values = []
                for k in range(n_signals):
                    mpc = phase.MPC(x_phase[:,k], y_phase[:,k])
                    mpc_values.append(mpc)

                MPC_matrix[i, j] = np.mean(mpc_values)

    return MPC_matrix, fc_array


def mpc_heatmap(MPC: np.ndarray,
                array: np.ndarray,
                log: bool = True,
                xlabel: str = "X bandwidth (Hz)",
                ylabel: str = "Y bandwidth (Hz)",
                title: str = "Mean Phase Coherence",
                cmap: str = "viridis",
                interpolation: str = "nearest"):

    plt.figure(figsize=(6, 5))

    if log:
        MPC = np.log(MPC)
        label = "log(MPC)"
    else:
        label = "MPC"

    im = plt.imshow(
        MPC,
        origin='lower',
        extent=[array[0], array[-1], array[0], array[-1]],
        aspect='auto',
        cmap=cmap,
        interpolation=interpolation
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label(label)

    plt.tight_layout()
    plt.show()


def mpc_surface(MPC: np.ndarray,
                array: np.ndarray,
                log: bool = False,
                xlabel: str = "X bandwidth (Hz)",
                ylabel: str = "Y bandwidth (Hz)",
                title: str = "MPC Surface Plot",
                cmap: str = "viridis"):
    
    X, Y = np.meshgrid(array, array)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if log:
        MPC = np.log(MPC)
        zlabel = "log(MPC)"
    else:
        zlabel = "MPC"

    surf = ax.plot_surface(X, Y, MPC, cmap=cmap, edgecolor='none', antialiased=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=10, label=zlabel)

    plt.tight_layout()
    plt.show()