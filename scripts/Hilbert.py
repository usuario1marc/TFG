import numpy as np
from scipy.integrate import solve_ivp           #https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
from scipy.signal import hilbert                #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
import matplotlib.pyplot as plt


def rossler(t, state, w=1):
    x1, x2, x3 = state
    dx1dt = -w*x2 - x3
    dx2dt = w*x1 + 0.165*x2
    dx3dt = 0.2 + x3 * (x1 - 10)
    return [dx1dt, dx2dt, dx3dt]

def coupled_rossler(t, state, wx=1, wy=1, ex=0, ey=0):
    x1, x2, x3, y1, y2, y3 = state

    dx1dt = -wx*x2 - x3 + ex*(y1-x1)
    dx2dt = wx*x1 + 0.165*x2
    dx3dt = 0.2 + x3 * (x1 - 10)

    dy1dt = -wy*y2 - y3 + ey*(x1-y1)
    dy2dt = wy*y1 + 0.165*y2
    dy3dt = 0.2 + y3 * (y1 - 10)

    return [dx1dt, dx2dt, dx3dt, dy1dt, dy2dt, dy3dt]

def euler(f, t_span: tuple, state_0: np.ndarray, dt: float, args: tuple = (), std: float = 0) -> np.ndarray:
    t_eval = np.arange(t_span[0], t_span[1], dt)
    state_vec = np.zeros((t_eval.size, state_0.size))
    state_vec[0, :] = state_0

    for i, t in enumerate(t_eval):
        if i==0:
            continue
        gradient = f(t, state_vec[i-1, :], *args)
        noise = np.random.random(len(state_0)) * np.array([1, 1, 0]*int(len(state_0)/3))
        state_vec[i, :] = state_vec[i-1, :] + np.array(gradient) * dt + noise * np.sqrt(dt) * std

    return state_vec


dt = 0.001
downsample = 300
total_points = 10000
t_span = (0, total_points * dt * downsample)    #TOT EL INTERVAL QUE VOL INTEGRAR       
t_eval = np.arange(0, t_span[1], dt)            #CADA QUAN INTEGRA
y0 = np.random.rand(3)

'''
sol = solve_ivp(coupled_rossler, t_span, y0, t_eval=t_eval, method='RK45', args=args)
x = sol.y[0]
y = sol.y[3]
'''

print("Integrating Rössler...")
args = (0.5, 1, 0.5, 0)
sol = euler(rossler, t_span, y0, dt, (1,), 0.25)
x = sol[:, 0]

print("Integrating Rössler...")
args = (0.5, 1, 0.5, 0)
sol = euler(rossler, t_span, y0, dt, (1,), 0)
y = sol[:, 0]

print("Done. Saving output...")

x_down = x[::downsample]
y_down = y[::downsample]
x_final = x_down[-4096:]

t_down = np.arange(len(x_down)) * (dt * downsample)
t_final = t_down[-4096:]  # matching last segment

#Save output
np.savez("noisy.npz", x=x_down, y=y_down, t=t_down, args=args)
print("Done.")

'''
#Hilbert transform
x_hilbert = hilbert(x_final)

signal = x_final + 1j * x_hilbert.imag
amplitude = np.abs(signal)
phase = np.unwrap(np.angle(signal))

plt.figure(figsize=(12, 8))

# --- Original signal ---
plt.subplot(3, 1, 1)
plt.plot(t_final, x_final, 'k', lw=1)
plt.title("Rössler system (x₁ component) and Hilbert Transform results")
plt.ylabel("x₁(t)")

# --- Instantaneous amplitude ---
plt.subplot(3, 1, 2)
plt.plot(t_final, amplitude, 'b')
plt.ylabel("Amplitude |X(t)|")

# --- Instantaneous phase ---
plt.subplot(3, 1, 3)
plt.plot(t_final, phase, 'r')
plt.ylabel("Phase φ(t)")
plt.xlabel("Time [units]")

plt.tight_layout()
plt.show()
'''