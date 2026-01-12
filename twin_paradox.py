import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# Twin Paradox Simulator (Special Relativity)
# Units: c = 1  -> v = beta
# Proper time update:
#   dτ = sqrt(1 - β^2) dt  = dt / γ
# =========================

# ---- Parameters (ปรับได้) ----
beta_max = 0.90     # max v/c (must be < 1)
t_accel = 10.0      # accelerate duration (Earth time)
t_cruise = 20.0     # cruise duration
t_decel = 10.0      # decelerate duration
dt = 0.02           # timestep (seconds in Earth frame)
speedup = 1         # animation steps per frame (increase if slow)

# Total time: outbound + inbound (mirror)
leg_T = t_accel + t_cruise + t_decel
T_total = 2 * leg_T

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def beta_profile(t_leg: float) -> float:
    """speed magnitude profile for one leg (OUT or BACK)"""
    if t_leg < 0:
        return 0.0
    if t_leg < t_accel:
        return beta_max * (t_leg / t_accel)
    t2 = t_leg - t_accel
    if t2 < t_cruise:
        return beta_max
    t3 = t2 - t_cruise
    if t3 < t_decel:
        return beta_max * (1.0 - t3 / t_decel)
    return 0.0

# ---- Precompute simulation arrays ----
N = int(np.ceil(T_total / dt)) + 1

t = np.zeros(N)        # Earth time
tau = np.zeros(N)      # ship proper time
x = np.zeros(N)        # ship position in Earth frame
beta = np.zeros(N)     # signed velocity (v/c)

for i in range(1, N):
    t[i] = t[i-1] + dt

    # Determine leg and time within leg
    if t[i] <= leg_T:
        direction = +1.0
        t_leg = t[i]
    else:
        direction = -1.0
        t_leg = t[i] - leg_T

    bmag = clamp(beta_profile(t_leg), 0.0, 0.999999)
    beta[i] = direction * bmag

    # Proper time increment: dτ = sqrt(1-β^2) dt
    tau[i] = tau[i-1] + np.sqrt(1.0 - beta[i]**2) * dt

    # Position increment: dx = v dt = β dt (since c=1)
    x[i] = x[i-1] + beta[i] * dt

# ---- Plot setup ----
plt.rcParams["figure.figsize"] = (11, 6)
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[0, 0])  # tau vs t
ax2 = fig.add_subplot(gs[0, 1])  # beta vs t
ax3 = fig.add_subplot(gs[1, :])  # x vs t (and "ship marker")

fig.suptitle("Twin Paradox Simulator (Special Relativity, c = 1)", fontsize=14)

# Ax1: tau(t)
ax1.set_title("Proper Time on Ship vs Earth Time")
ax1.set_xlabel("Earth time t (s)")
ax1.set_ylabel("Ship proper time τ (s)")
ax1.set_xlim(0, T_total)
ax1.set_ylim(0, max(tau) * 1.05)
line_tau, = ax1.plot([], [], lw=2)
dot_tau, = ax1.plot([], [], marker="o")

# Ax2: beta(t)
ax2.set_title("Velocity Profile β(t) = v/c")
ax2.set_xlabel("Earth time t (s)")
ax2.set_ylabel("β")
ax2.set_xlim(0, T_total)
ax2.set_ylim(-1.0, 1.0)
line_beta, = ax2.plot([], [], lw=2)
dot_beta, = ax2.plot([], [], marker="o")

# Ax3: x(t) and a marker
ax3.set_title("Ship Position x(t) in Earth Frame")
ax3.set_xlabel("Earth time t (s)")
ax3.set_ylabel("x (light-seconds, since c=1)")
ax3.set_xlim(0, T_total)
xmin, xmax = min(x), max(x)
pad = (xmax - xmin) * 0.15 + 1e-6
ax3.set_ylim(xmin - pad, xmax + pad)
line_x, = ax3.plot([], [], lw=2)
dot_x, = ax3.plot([], [], marker="o")

# Text box summary
text_box = ax3.text(
    0.02, 0.95, "", transform=ax3.transAxes, va="top",
    bbox=dict(boxstyle="round", alpha=0.2)
)

def init():
    line_tau.set_data([], [])
    dot_tau.set_data([], [])
    line_beta.set_data([], [])
    dot_beta.set_data([], [])
    line_x.set_data([], [])
    dot_x.set_data([], [])
    text_box.set_text("")
    return line_tau, dot_tau, line_beta, dot_beta, line_x, dot_x, text_box

def update(frame):
    # Map animation frame -> index i
    i = min(frame * speedup, N - 1)

    # Update τ(t)
    line_tau.set_data(t[:i+1], tau[:i+1])
    dot_tau.set_data([t[i]], [tau[i]])

    # Update β(t)
    line_beta.set_data(t[:i+1], beta[:i+1])
    dot_beta.set_data([t[i]], [beta[i]])

    # Update x(t)
    line_x.set_data(t[:i+1], x[:i+1])
    dot_x.set_data([t[i]], [x[i]])

    # Compute gamma (instantaneous)
    b = abs(beta[i])
    gamma = 1.0 / np.sqrt(1.0 - b*b) if b < 1 else np.inf

    # Leg label
    leg = "OUTBOUND" if t[i] <= leg_T else "INBOUND"

    # Summary
    diff = t[i] - tau[i]
    text_box.set_text(
        f"Leg: {leg}\n"
        f"t = {t[i]:.2f} s\n"
        f"τ = {tau[i]:.2f} s\n"
        f"t - τ = {diff:.2f} s\n"
        f"β = {beta[i]:+.3f}\n"
        f"γ = {gamma:.3f}\n"
        f"x = {x[i]:+.2f}"
    )

    return line_tau, dot_tau, line_beta, dot_beta, line_x, dot_x, text_box

frames = int(np.ceil(N / speedup))
ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=20, blit=True)

# ---- Final result print when window closes ----
def on_close(event):
    total_diff = t[-1] - tau[-1]
    print("\n=== Twin Paradox Result (Trip Complete) ===")
    print(f"Earth elapsed time t  = {t[-1]:.3f} s")
    print(f"Ship elapsed time  τ  = {tau[-1]:.3f} s")
    print(f"Difference (t - τ)    = {total_diff:.3f} s")
    print("=========================================\n")

fig.canvas.mpl_connect("close_event", on_close)

plt.show()
