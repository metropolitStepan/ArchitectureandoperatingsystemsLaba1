import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

t_c = 1e-3
h   = 1e-6
t = np.arange(0, t_c + h/2, h)
t_ms = t * 1e3

U0, U1 = 0.5, 4.5
f5, f10 = 5e3, 10e3
T1, T2 = 1e-5, 2e-5
A1, A2 = 0.1, 0.3

# Функции
def rect_pulse(f, t, U0, U1):
    T = 1.0 / f
    return np.where((t % T) < (T / 2), U1, U0)

def rc_input_response(Y, h, T, U_init=None):
    U = np.zeros_like(Y, dtype=float)
    if U_init is None:
        U_init = Y[0]
    U[0] = U_init
    alpha = h / T
    for n in range(len(Y) - 1):
        U[n+1] = U[n] + alpha * (Y[n] - U[n])
    return U

def add_noise(U, A, rng):
    return U + rng.uniform(-A, A, size=U.shape)

#Сигналы
Y5, Y10 = rect_pulse(f5, t, U0, U1), rect_pulse(f10, t, U0, U1)

U5_T1   = rc_input_response(Y5,  h, T1)
U5_T2   = rc_input_response(Y5,  h, T2)
U10_T1  = rc_input_response(Y10, h, T1)
U10_T2  = rc_input_response(Y10, h, T2)

rng = np.random.default_rng(0)
U5_T1_A1  = add_noise(U5_T1,  A1, rng); U5_T1_A2  = add_noise(U5_T1,  A2, rng)
U5_T2_A1  = add_noise(U5_T2,  A1, rng); U5_T2_A2  = add_noise(U5_T2,  A2, rng)
U10_T1_A1 = add_noise(U10_T1, A1, rng); U10_T1_A2 = add_noise(U10_T1, A2, rng)
U10_T2_A1 = add_noise(U10_T2, A1, rng); U10_T2_A2 = add_noise(U10_T2, A2, rng)

COLOR_Y   = 'tab:orange'  # вход Y
COLOR_U   = 'tab:blue'    # U(t) без помех
COLOR_A1  = 'tab:green'   # U(t) с помехой A=0.1 В
COLOR_A2  = 'tab:red'     # U(t) с помехой A=0.3 В

fig, axes = plt.subplots(14, 1, figsize=(12, 28), sharex=True)

def setup(ax, title):
    ax.set_title(title)
    ax.set_ylabel("Напряжение, В")
    ax.grid(True)
    ax.set_ylim(0, 5)

# Пункт 1
axes[0].plot(t_ms, Y5,  color=COLOR_Y,  linewidth=1.4)
setup(axes[0], "П.1 — Сигнал генератора, 5 кГц (только вход)")
axes[1].plot(t_ms, Y10, color=COLOR_Y,  linewidth=1.4)
setup(axes[1], "П.1 — Сигнал генератора, 10 кГц (только вход)")

# Пункт 2
axes[2].plot(t_ms, Y5,  color=COLOR_Y, linewidth=1.2)
axes[2].plot(t_ms, U5_T1, color=COLOR_U, linewidth=1.2, linestyle='--')
setup(axes[2], "П.2 — 5 кГц: Y и U(t), T = 10 мкс")

axes[3].plot(t_ms, Y5,  color=COLOR_Y, linewidth=1.2)
axes[3].plot(t_ms, U5_T2, color=COLOR_U, linewidth=1.2, linestyle='--')
setup(axes[3], "П.2 — 5 кГц: Y и U(t), T = 20 мкс")

axes[4].plot(t_ms, Y10, color=COLOR_Y, linewidth=1.2)
axes[4].plot(t_ms, U10_T1, color=COLOR_U, linewidth=1.2, linestyle='--')
setup(axes[4], "П.2 — 10 кГц: Y и U(t), T = 10 мкс")

axes[5].plot(t_ms, Y10, color=COLOR_Y, linewidth=1.2)
axes[5].plot(t_ms, U10_T2, color=COLOR_U, linewidth=1.2, linestyle='--')
setup(axes[5], "П.2 — 10 кГц: Y и U(t), T = 20 мкс")

# Пункт 3
axes[6].plot(t_ms, Y5,      color=COLOR_Y, linewidth=1.0)
axes[6].plot(t_ms, U5_T1_A1, color=COLOR_A1, linewidth=1.0, linestyle='--')
setup(axes[6], "П.3 — 5 кГц: T = 10 мкс, помеха A = 0,1 В")

axes[7].plot(t_ms, Y5,      color=COLOR_Y, linewidth=1.0)
axes[7].plot(t_ms, U5_T1_A2, color=COLOR_A2, linewidth=1.0, linestyle='--')
setup(axes[7], "П.3 — 5 кГц: T = 10 мкс, помеха A = 0,3 В")

axes[8].plot(t_ms, Y5,      color=COLOR_Y, linewidth=1.0)
axes[8].plot(t_ms, U5_T2_A1, color=COLOR_A1, linewidth=1.0, linestyle='--')
setup(axes[8], "П.3 — 5 кГц: T = 20 мкс, помеха A = 0,1 В")

axes[9].plot(t_ms, Y5,      color=COLOR_Y, linewidth=1.0)
axes[9].plot(t_ms, U5_T2_A2, color=COLOR_A2, linewidth=1.0, linestyle='--')
setup(axes[9], "П.3 — 5 кГц: T = 20 мкс, помеха A = 0,3 В")

axes[10].plot(t_ms, Y10,       color=COLOR_Y, linewidth=1.0)
axes[10].plot(t_ms, U10_T1_A1, color=COLOR_A1, linewidth=1.0, linestyle='--')
setup(axes[10], "П.3 — 10 кГц: T = 10 мкс, помеха A = 0,1 В")

axes[11].plot(t_ms, Y10,       color=COLOR_Y, linewidth=1.0)
axes[11].plot(t_ms, U10_T1_A2, color=COLOR_A2, linewidth=1.0, linestyle='--')
setup(axes[11], "П.3 — 10 кГц: T = 10 мкс, помеха A = 0,3 В")

axes[12].plot(t_ms, Y10,       color=COLOR_Y, linewidth=1.0)
axes[12].plot(t_ms, U10_T2_A1, color=COLOR_A1, linewidth=1.0, linestyle='--')
setup(axes[12], "П.3 — 10 кГц: T = 20 мкс, помеха A = 0,1 В")

axes[13].plot(t_ms, Y10,       color=COLOR_Y, linewidth=1.0)
axes[13].plot(t_ms, U10_T2_A2, color=COLOR_A2, linewidth=1.0, linestyle='--')
setup(axes[13], "П.3 — 10 кГц: T = 20 мкс, помеха A = 0,3 В")

axes[-1].set_xlabel("Время, мс")


legend_handles = [
    Line2D([0], [0], color=COLOR_Y,  lw=2, label="Y — сигнал генератора (вход)"),
    Line2D([0], [0], color=COLOR_U,  lw=2, ls='--', label="U(t) — вход каскада (без помех)"),
    Line2D([0], [0], color=COLOR_A1, lw=2, ls='--', label="U(t) — с помехой A = 0,1 В"),
    Line2D([0], [0], color=COLOR_A2, lw=2, ls='--', label="U(t) — с помехой A = 0,3 В"),
]
fig.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(0.82, 0.5),
           title="Обозначения", frameon=True)

plt.subplots_adjust(right=0.78, hspace=0.6)
plt.show()
