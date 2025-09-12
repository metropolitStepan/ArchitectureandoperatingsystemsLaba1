import numpy as np
import matplotlib.pyplot as plt

h = 1e-6
t_end = 1e-3
x = np.arange(0.0, t_end, h)

U0, U1 = 0.5, 4.5
A1, A2 = 0.1, 0.3
T_list = [10e-6, 20e-6]
f_list = [5e3, 10e3]


def gen_y_rect(f_hz, x, U0, U1):
    T = 1.0 / f_hz
    phase = (x % T) / T
    y = np.where(phase < 0.5, U0, U1).astype(float)
    return y

def rc_response(y, h, T):
    u = np.zeros_like(y, dtype=float)
    u[0] = y[0]
    alpha = h / T
    for i in range(0, len(y) - 1):
        u[i + 1] = u[i] + alpha * (y[i] - u[i])
    return u

def add_noise(u, A, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    return u + (2*A * rng.random(size=u.shape) - A)

def logic_output_schmitt(u):
    v = np.zeros_like(u, dtype=float)
    v[0] = 0.5
    for i in range(0, len(u) - 1):
        if (u[i] < 4.0) and (u[i + 1] > 4.0):
            v[i + 1] = 4.5
        elif (u[i] > 2.0) and (u[i + 1] < 2.0):
            v[i + 1] = 0.5
        else:
            v[i + 1] = v[i]
    return v

fig, axes = plt.subplots(8, 1, figsize=(10, 18), sharex=True)

row = 0
rng = np.random.default_rng(0)

for f in f_list:
    y = gen_y_rect(f, x, U0, U1)

    for T in T_list:
        u_clean = rc_response(y, h, T)

        for A in (A1, A2):
            u_noisy = add_noise(u_clean, A, rng=rng)

            v = logic_output_schmitt(u_noisy)

            ax = axes[row]
            ax.plot(x*1e3, y, lw=1.0)
            ax.plot(x*1e3, v, lw=1.2)
            ax.set_title(f"П.4 — Выход каскада | f = {int(f/1e3)} кГц, T = {int(T*1e6)} мкс, A = {A} В")
            ax.set_ylabel("Напряжение, В")
            ax.grid(True)
            row += 1

axes[-1].set_xlabel("Время, мс")
plt.tight_layout()
plt.show()
