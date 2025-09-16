import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from utils.GetClassicOrbitalElements import *
from utils.visualization import plot_classic_orbital_elements


r = np.array([10016.34, -17012.52, 7899.28])
v = np.array([2.5, -1.05, 3.88])
t = np.linspace(0, 432000, 100000)
earth_radius = 6378.0  # in km
mu = 3.986e5
thrust = 0.01

# ===================== ADIÇÕES (massa e Busek BIT-3) =====================
# Dados do BIT-3
T   = 0.01   # N (força fixa)
Isp = 2150.0       # s
g0  = 9.80665      # m/s^2

# Estado de massa
m_sat = 20.0       # kg (massa inicial nominal usada no seu aV/aH)
m0    = 20.0       # kg (massa inicial do estado)
m_dry = 15.0       # kg (massa seca)
# ========================================================================

# ===================== ADIÇÕES (H e janelas em anomalia verdadeira) =====================
# Aqui não importa mais a janela, já que não usamos H — mas mantive a estrutura
THRUST_INTERVAL_DEG = 30.0
MEAN_THETA_LIST_DEG = [180.0]

def throttle(t, x):
    return 1.0 if x[6] > m_dry else 0.0

def wrap_deg(a):
    return np.remainder(a, 360.0)

def angle_in_window_deg(theta_deg, center_deg, width_deg):
    half = 0.5*width_deg
    lo = wrap_deg(center_deg - half)
    hi = wrap_deg(center_deg + half)
    th = wrap_deg(theta_deg)
    if lo <= hi:
        return (th >= lo) and (th <= hi)
    else:
        return (th >= lo) or (th <= hi)

def in_any_window(theta_deg):
    return any(angle_in_window_deg(theta_deg, cdeg, THRUST_INTERVAL_DEG)
               for cdeg in MEAN_THETA_LIST_DEG)
# ========================================================================================

def x_dot(t, x):
    xdot = np.zeros_like(x)
    xdot[0] = x[3]
    xdot[1] = x[4]
    xdot[2] = x[5]

    # Gravidade
    r_vec = x[0:3]
    v_vec = x[3:6]
    rnorm = np.linalg.norm(r_vec)
    xdot[3:6] = -(mu/(rnorm**3))*r_vec

    # ---------- Propulsão só em V ----------
    m_cur = max(x[6], 1e-18)
    u = throttle(t, x)
    a_inst = (T / m_cur) / 1000.0

    r_hat = r_vec / (np.linalg.norm(r_vec) + 1e-32)
    h_vec_local = np.cross(r_vec, v_vec)
    w_hat = h_vec_local / (np.linalg.norm(h_vec_local) + 1e-32)
    s_hat = np.cross(w_hat, r_hat)
    s_hat /= (np.linalg.norm(s_hat) + 1e-32)

    dir_hat = s_hat  # apenas V

    if u > 0.0:
        xdot[3:6] += a_inst * dir_hat
        xdot[6] = - T/(Isp*g0)
    else:
        xdot[6] = 0.0
    # ---------------------------------------

    return xdot

# >>> Estado inicial inclui massa <<<
x0 = np.concatenate((r, v, [m0]))

sol = solve_ivp(x_dot, (t[0], t[-1]), x0, t_eval=t, method='RK45')

X = sol.y

# ---------- elementos, ν (0–360) e i ----------
orbital_elementss = []
nus_deg = []
incs_deg = []
for k in range(X.shape[1]):
    r_vec = X[0:3, k]
    v_vec = X[3:6, k]
    orbital_elementss.append(get_orbital_elements(r_vec, v_vec, mu))
    nu = get_true_anormaly(r_vec, v_vec, mu)
    if np.dot(r_vec, v_vec) < 0.0:
        nu = (360.0 - nu) % 360.0
    nus_deg.append(nu)
    incs_deg.append(get_inclination(r_vec, v_vec, mu))

nus_deg = np.array(nus_deg)
incs_deg = np.array(incs_deg)

# ---------- Δv numérico ----------
tt = sol.t
r_norm_series = np.linalg.norm(X[0:3, :].T, axis=1)
v_norm_series = np.linalg.norm(X[3:6, :].T, axis=1)
m_series       = X[6, :]

dt = np.diff(tt)
a_inst_series = (T / np.maximum(m_series, 1e-18)) / 1000.0
delta_v_kms = float(np.sum(a_inst_series[:-1] * dt * (m_series[:-1] > m_dry)))
delta_v_ms  = 1000.0 * delta_v_kms

print("\n=== Satélite V-only ===")
print(f"Δv acumulado (m/s):        {delta_v_ms:.6f}")
print(f"Δi sim (último - inicial): {incs_deg[-1] - incs_deg[0]:.9f}")

def simulate():
    # >>> Estado inicial inclui massa <<<
    x0 = np.concatenate((r, v, [m0]))

    sol = solve_ivp(x_dot, (t[0], t[-1]), x0, t_eval=t, method='RK45')

    X = sol.y

    orbital_elementss = []
    nus_deg = []
    incs_deg = []
    for k in range(X.shape[1]):
        r_vec = X[0:3, k]
        v_vec = X[3:6, k]
        orbital_elementss.append(get_orbital_elements(r_vec, v_vec, mu))
        nu = get_true_anormaly(r_vec, v_vec, mu)
        if np.dot(r_vec, v_vec) < 0.0:
            nu = (360.0 - nu) % 360.0
        nus_deg.append(nu)
        incs_deg.append(get_inclination(r_vec, v_vec, mu))

    return t, X, np.array(nus_deg), np.array(incs_deg), orbital_elementss



plot_classic_orbital_elements(t, orbital_elementss)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
earth_radius = 6378.0  # km

# Wireframe da Terra
u, vgrid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = earth_radius * np.cos(u) * np.sin(vgrid)
y_e = earth_radius * np.sin(u) * np.sin(vgrid)
z_e = earth_radius * np.cos(vgrid)
ax.plot_wireframe(x_e, y_e, z_e, color="g", alpha=0.3)

# Trajetória simulada (única)
ax.plot3D(X[0, :], X[1, :], X[2, :], 'b-', label="Satélite V-only")

# Ajustes visuais
ax.set_box_aspect([1, 1, 1])
ax.set_title("Órbita simulada - Satélite V-only")
ax.legend()
plt.show()