import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from utils.GetClassicOrbitalElements import *
from utils.visualization import plot_classic_orbital_elements
import matplotlib.patches as mpatches

# ===================== condições iniciais =====================
r = np.array([10016.34, -17012.52, 7899.28])
v = np.array([2.5, -1.05, 3.88])
t = np.linspace(0, 4320000, 1000000)  # 50 dias, 1e6 pontos
earth_radius = 6378.0  # km
mu = 3.986e5

# ===================== Dados de propulsão =====================
T   = 1.1e-3       # N
Isp = 2150.0       # s
g0  = 9.80665      # m/s^2
m0  = 20.0         # kg
m_dry = 15.0       # kg

# ===== Controle de tolerâncias (porcentagem de erro) =====
ERROR_PERCENT = 0.9               # << ajuste aqui (%, p.ex. 1.0 = 1%)
RTOL = ERROR_PERCENT / 100.0        # erro relativo alvo
# ATOL por estado [x,y,z,vx,vy,vz,m] — ajuste às suas unidades/escala:
ATOL = np.array([1e-3, 1e-3, 1e-3,   # posição em km (1e-3 km = 1 m)
                 1e-6, 1e-6, 1e-6,   # velocidade em km/s (1e-6 km/s = 1 mm/s)
                 1e-6])              # massa em kg

# adição da perturbação
from J2 import external_accel, EarthParams, PerturbationFlags

# Flag de módulo para ligar/desligar J2 (não mexe no resto do código)
_J2_ON = False

def _accel_J2(r_vec, v_vec, tnow):
    if _J2_ON:
        return external_accel(r_vec, v_vec, tnow,
                              params=EarthParams(),
                              flags=PerturbationFlags(j2=True))
    return 0.0 * r_vec

# definição do inclinação para 97.5 do ITASAT
def _R3(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def _R1(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  c, -s ],
                     [0.0,  s,  c ]])

def _coe_to_rv(a, e, i_deg, raan_deg, argp_deg, nu_deg, mu):
    i  = np.deg2rad(i_deg)
    O  = np.deg2rad(raan_deg)   # RAAN (Ω)
    w  = np.deg2rad(argp_deg)   # argumento do perigeu (ω)
    nu = np.deg2rad(nu_deg)     # anomalia verdadeira (ν)

    p = a * (1.0 - e**2)
    r_pqw = np.array([
        p * np.cos(nu) / (1.0 + e*np.cos(nu)),
        p * np.sin(nu) / (1.0 + e*np.cos(nu)),
        0.0
    ])
    v_pqw = np.array([
        -np.sqrt(mu/p) * np.sin(nu),
        +np.sqrt(mu/p) * (e + np.cos(nu)),
        0.0
    ])

    C = _R3(O) @ _R1(i) @ _R3(w)
    return C @ r_pqw, C @ v_pqw

def reseed_state_with_inclination(r, v, mu, i_target_deg):
    a0  = get_major_axis(r, v, mu)
    e0  = get_eccentricity(r, v, mu)
    O0  = get_ascending_node(r, v, mu)
    w0  = get_argument_of_perigee(r, v, mu)
    nu0 = get_true_anormaly(r, v, mu)   # [0,360)
    return _coe_to_rv(a0, e0, i_target_deg, O0, w0, nu0, mu)

# ======== alvo de inclinação inicial (em graus) ========
I0_TARGET_DEG = 97.5
r, v = reseed_state_with_inclination(r, v, mu, I0_TARGET_DEG)
print(f"[check] i0 = {get_inclination(r, v, mu):.6f} deg")

# Dois motores independentes (V e H)
DUAL_THRUSTERS = True
THRUST_MODE = "sum"  # "sum" ou "split"
if DUAL_THRUSTERS:
    if THRUST_MODE == "sum":
        T_V, T_H = T, T
    elif THRUST_MODE == "split":
        T_V, T_H = 0.5*T, 0.5*T
    else:
        raise ValueError("THRUST_MODE deve ser 'sum' ou 'split'.")
    Isp_V, Isp_H = Isp, Isp

# ===================== Janelas em anomalia verdadeira =====================
THRUST_INTERVAL_DEG = 30.0
MEAN_THETA_LIST_DEG = [180.0]  # apogeu

def throttle(tnow, x):
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

# ===================== Dinâmica =====================
def x_dot(tnow, x):
    xdot = np.zeros_like(x)
    xdot[0:3] = x[3:6]

    r_vec = x[0:3]
    v_vec = x[3:6]
    rnorm = np.linalg.norm(r_vec) + 1e-32

    # Gravidade
    xdot[3:6] = -(mu/(rnorm**3))*r_vec

    # Perturbação J2 (se ligada)
    xdot[3:6] += _accel_J2(r_vec, v_vec, tnow)

    # Base RSW
    r_hat = r_vec / rnorm
    h_vec_local = np.cross(r_vec, v_vec)
    w_hat = h_vec_local / (np.linalg.norm(h_vec_local) + 1e-32)
    s_hat = np.cross(w_hat, r_hat)
    s_hat /= (np.linalg.norm(s_hat) + 1e-32)

    # Janela de H por nu
    theta_deg = get_true_anormaly(r_vec, v_vec, mu)
    fire_H = in_any_window(theta_deg)

    # Empuxo
    m_cur = max(x[6], 1e-18)
    u = throttle(tnow, x)

    if not DUAL_THRUSTERS:
        a_inst = (T / m_cur) / 1000.0
        alpha = np.deg2rad(45.0) if fire_H else 0.0
        dir_hat = np.cos(alpha)*s_hat - np.sin(alpha)*w_hat  # DOWN
        if u > 0.0:
            xdot[3:6] += a_inst * dir_hat
            xdot[6] = - T/(Isp*g0)
        else:
            xdot[6] = 0.0
    else:
        aV = (T_V / m_cur) / 1000.0
        aH = (T_H / m_cur) / 1000.0 if fire_H else 0.0
        SIGN_H = -1.0  # DOWN
        if u > 0.0:
            xdot[3:6] += aV * s_hat + SIGN_H * aH * w_hat
            mdot_V = T_V/(Isp_V*g0)
            mdot_H = (T_H/(Isp_H*g0)) if fire_H else 0.0
            xdot[6] = - (mdot_V + mdot_H)
        else:
            xdot[6] = 0.0

    return xdot

# >>> estado inicial (inclui massa) <<<
x0 = np.concatenate((r, v, [m0]))

# ====== usar dense_output e tolerâncias (RTOL/ATOL) ======
sol = solve_ivp(x_dot, (t[0], t[-1]), x0,
                method='DOP853',        # ou 'RK45'; DOP853 costuma ser mais eficiente
                rtol=RTOL, atol=ATOL,
                dense_output=True)
X = sol.sol(t)  # X tem shape (7, len(t))

# ---------- elementos, nu (0–360) e i ----------
orbital_elementss = []
nus_deg = []
incs_deg = []
for k in range(X.shape[1]):
    r_vec = X[0:3, k]
    v_vec = X[3:6, k]
    orbital_elementss.append(get_orbital_elements(r_vec, v_vec, mu))
    nu = get_true_anormaly(r_vec, v_vec, mu)  # [0, 360)
    nus_deg.append(nu)
    incs_deg.append(get_inclination(r_vec, v_vec, mu))

nus_deg = np.array(nus_deg)
incs_deg = np.array(incs_deg)

# ---------- métricas ----------
tt = t
r_norm_series = np.linalg.norm(X[0:3, :].T, axis=1)
v_norm_series = np.linalg.norm(X[3:6, :].T, axis=1)
m_series = X[6, :]

fire_mask = np.array([in_any_window(nu) for nu in nus_deg], dtype=bool)
u_mask = (m_series > m_dry)

dt = np.diff(tt)
aV_series = (T_V / np.maximum(m_series, 1e-18)) / 1000.0
aH_series = (T_H / np.maximum(m_series, 1e-18)) / 1000.0

delta_v_H_kms = float(np.sum(aH_series[:-1] * dt * (fire_mask[:-1] & u_mask[:-1])))
delta_v_H_ms  = 1000.0 * delta_v_H_kms
t_H_on = float(np.sum(dt * fire_mask[:-1]))

if np.any(fire_mask):
    v_center = float(np.mean(v_norm_series[fire_mask]))
else:
    v_center = float(np.min(v_norm_series))  # fallback

arg = np.clip(delta_v_H_kms/(2.0*v_center), -1.0, 1.0)
delta_i_ideal_deg = float(np.degrees(2.0*np.arcsin(arg)))
delta_i_sim_deg = incs_deg - incs_deg[0]

print("\n=== Dados V H Down ===")
print(f"Tempo com H ligado (s):     {t_H_on:.6f}")
print(f"Δv_H acumulado (m/s):       {delta_v_H_ms:.6f}")
print(f"Δi_ideal (graus):           {delta_i_ideal_deg:.9f}")
print(f"Δi_sim (último - inicial):  {delta_i_sim_deg[-1]:.9f}")

# ---------- plot dos elementos ----------
plot_classic_orbital_elements(t, orbital_elementss)

# 1) máscara "empuxo H ON": janela em nu E massa > m_dry
thr_H_on_mask = (m_series > m_dry) & np.array([in_any_window(nu) for nu in nus_deg], dtype=bool)

# 2) utilitários para converter máscara em intervalos e sombrear no gráfico
def _mask_to_spans(tgrid, mask_bool):
    tgrid = np.asarray(tgrid, float); mask = np.asarray(mask_bool, bool)
    if mask.size == 0: return []
    rises  = np.where(np.diff(mask.astype(np.int8)) == +1)[0] + 1
    falls  = np.where(np.diff(mask.astype(np.int8)) == -1)[0] + 1
    if mask[0]:  rises = np.r_[0, rises]
    if mask[-1]: falls = np.r_[falls, mask.size-1]
    return list(zip(tgrid[rises], tgrid[falls]))

def add_thrust_spans(ax, tgrid, mask_bool, *, color="tab:orange", alpha=0.15, label="Empuxo H ON"):
    for t0, t1 in _mask_to_spans(tgrid, mask_bool):
        ax.axvspan(t0, t1, color=color, alpha=alpha, lw=0)
    handles, labels = ax.get_legend_handles_labels()
    patch = mpatches.Patch(color=color, alpha=alpha, label=label)
    if label not in labels:
        ax.legend(handles + [patch], labels + [label])

def add_thrust_edges(ax, tgrid, mask_bool, *, color="tab:orange", alpha=0.7):
    idx_on = np.where(np.diff(mask_bool.astype(np.int8)) == +1)[0] + 1
    for i in idx_on:
        ax.axvline(tgrid[i], color=color, ls="--", lw=0.8, alpha=alpha)

# 3) Ω(t) desenrolado com faixas do empuxo H
def _unwrap_deg(a_deg):
    return np.degrees(np.unwrap(np.radians(np.asarray(a_deg, float))))

Omega_series = [el.ascending_node for el in orbital_elementss]
Omega_unw = _unwrap_deg(Omega_series)

t_plot = t/86400.0  # em dias
fig, ax = plt.subplots()
ax.plot(t_plot, Omega_unw, 'r-', lw=1.2, label='Ω (desenrolado)')
add_thrust_spans(ax, t_plot, thr_H_on_mask, color="tab:orange", alpha=0.18, label="Empuxo H ON")
ax.set_xlabel("Tempo [dias]"); ax.set_ylabel("Ω [graus]")
ax.set_title("RAAN com janelas de empuxo H destacadas")
ax.grid(True); plt.show()

# ---------- plot 3D ----------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
u_grid, v_grid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = earth_radius * np.cos(u_grid) * np.sin(v_grid)
y_e = earth_radius * np.sin(u_grid) * np.sin(v_grid)
z_e = earth_radius * np.cos(v_grid)
ax.plot_wireframe(x_e, y_e, z_e, color="g", alpha=0.3)
ax.plot3D(X[0, :], X[1, :], X[2, :], 'r', label="Satélite V_H DOWN")
ax.set_box_aspect([1, 1, 1])
ax.set_title("Órbita simulada - Satélite V_H DOWN")
ax.legend()
ax.axis('equal')

# ---------- plot i(nu) segmentado ----------
def plot_i_vs_nu_segmentado(nu_deg: np.ndarray, inc_deg: np.ndarray, *, ax=None, **plot_kw):
    nu_deg = np.asarray(nu_deg, dtype=float)
    inc_deg = np.asarray(inc_deg, dtype=float)
    if ax is None:
        fig_loc, ax = plt.subplots()
    dn = np.diff(nu_deg)
    wraps = np.where(dn < -180.0)[0]
    start = 0
    for w in wraps:
        ax.plot(nu_deg[start:w+1], inc_deg[start:w+1], **plot_kw)
        start = w + 1
    ax.plot(nu_deg[start:], inc_deg[start:], **plot_kw)
    ax.set_xlabel(r'$\nu$ (deg)')
    ax.set_ylabel(r'$i$ (deg)')
    ax.set_xlim(0.0, 360.0)
    ax.grid(True)
    return ax

fig2, ax2 = plt.subplots()
plot_i_vs_nu_segmentado(nus_deg, incs_deg, ax=ax2, color="red", lw=1.5, label="i vs. nu")
ax2.legend()
plt.show()

def simulate():
    """
    Executa a integração com o x_dot deste módulo (VH DOWN) e devolve:
    (t, X, nus_deg, incs_deg, orbital_elementss)
    """
    x0 = np.concatenate((r, v, [m0]))
    sol = solve_ivp(x_dot, (t[0], t[-1]), x0,
                    method='DOP853', rtol=RTOL, atol=ATOL,
                    dense_output=True)
    Xloc = sol.sol(t)

    orbital_elementss = []
    nus_deg = []
    incs_deg = []

    for k in range(Xloc.shape[1]):
        r_vec = Xloc[0:3, k]
        v_vec = Xloc[3:6, k]
        orbital_elementss.append(get_orbital_elements(r_vec, v_vec, mu))
        nu = get_true_anormaly(r_vec, v_vec, mu)
        nus_deg.append(nu)
        incs_deg.append(get_inclination(r_vec, v_vec, mu))

    return t, Xloc, np.array(nus_deg, dtype=float), np.array(incs_deg, dtype=float), orbital_elementss
