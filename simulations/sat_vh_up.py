import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from utils.orbitalElementsOperations import*
from utils.visualization import plot_classic_orbital_elements
import matplotlib.patches as mpatches

# ===================== condições iniciais =====================
r = np.array([2128.5832, -6551.1055, 0.0000])   # km
v = np.array([-0.932454, -0.302973, 7.548972])
#r = np.array([6890.3, 0, 0]) #parametro da ITASAT 1
#v = np.array([0, -0.992,7.535])

#r = np.array([10016.34, -17012.52, 7899.28]) parametros orbitais iniciais
#v = np.array([2.5, -1.05, 3.88])
t = np.linspace(0, 432000, 1000000)  # 5 dias, 100k pontos
earth_radius = 6378.0  # km
mu = 3.986e5

# ===================== Dados de propulsão =====================
T   = 1.1e-3       # N
Isp = 2150.0       # s
g0  = 9.80665      # m/s^2
m0  = 20.0         # kg
m_dry = 15.0       # kg

# ===================== Perturbações (J2 hook) =====================
# Se seu módulo é simulations/perturbations.py, troque este import:
# from simulations.perturbations import external_accel, EarthParams, PerturbationFlags
from J2 import external_accel, EarthParams, PerturbationFlags

_J2_ON = False  # liga/desliga J2

def _accel_J2(r_vec, v_vec, t):
    if _J2_ON:
        return external_accel(r_vec, v_vec, t,
                              params=EarthParams(),
                              flags=PerturbationFlags(j2=True))
    return 0.0 * r_vec

#Adicionar arrasto atmosférico
from Drag import accel_drag, DragParams
_DRAG_ON = False
_DRAG = DragParams(Cd=2.2, A_ref_m2=0.02, use_atmo_rotation=True,
                   rho0_kg_m3=3.614e-11, h0_km=200.0, H_km=50.0)

def _accel_DRAG(r_vec, v_vec, m_cur):
    return accel_drag(r_vec, v_vec, m_cur, _DRAG) if _DRAG_ON else 0.0 * r_vec


# ===================== Dois motores independentes (V e H) =====================
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
MEAN_THETA_LIST_DEG = [180.0]  # 180 apogeu - 0 perigeu

def throttle(t, x):
    # Se não há empuxo, nunca liga
    if T <= 0.0:
        return 0.0
    else:
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
def x_dot(t, x):
    xdot = np.zeros_like(x)
    xdot[0:3] = x[3:6]

    r_vec = x[0:3]
    v_vec = x[3:6]
    rnorm = np.linalg.norm(r_vec) + 1e-32

    # Gravidade
    xdot[3:6] = -(mu/(rnorm**3))*r_vec

    # Perturbação J2 (opcional)
    xdot[3:6] += _accel_J2(r_vec, v_vec, t)

    # Perturbação arrasto atmosférico
    m_cur = max(x[6], 1e-18)
    xdot[3:6] += _accel_DRAG(r_vec, v_vec, m_cur)

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
    u = throttle(t, x)

    if not DUAL_THRUSTERS:
        a_inst = (T / m_cur) / 1000.0
        alpha = np.deg2rad(45.0) if fire_H else 0.0
        dir_hat = np.cos(alpha)*s_hat + np.sin(alpha)*w_hat  # UP
        if u > 0.0:
            xdot[3:6] += a_inst * dir_hat
            xdot[6] = - T/(Isp*g0)
        else:
            xdot[6] = 0.0
    else:
        aV = (T_V / m_cur) / 1000.0
        aH = (T_H / m_cur) / 1000.0 if fire_H else 0.0
        u_deg = get_argument_of_latitude(r_vec, v_vec, mu)     # u em graus
        SIGN_H = np.sign(np.cos(np.deg2rad(u_deg)))       

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
sol = solve_ivp(
    x_dot, (t[0], t[-1]), x0, t_eval=t,
    method="DOP853",       # ou "Radau" (implícito) se preferir
    rtol=1e-12, atol=1e-15,
)
X = sol.y

# ---------- elementos, nu (0–360) e i ----------
orbital_elementss = []
nus_deg = []
incs_deg = []
for k in range(X.shape[1]):
    r_vec = X[0:3, k]
    v_vec = X[3:6, k]
    orbital_elementss.append(get_orbital_elements(r_vec, v_vec, mu))
    nu = get_true_anormaly(r_vec, v_vec, mu)   # já em [0,360)
    nus_deg.append(nu)
    incs_deg.append(get_inclination(r_vec, v_vec, mu))

nus_deg = np.array(nus_deg, dtype=float)
incs_deg = np.array(incs_deg, dtype=float)

# ===================== MÉTRICAS (Δv, v_peri/apo, etc.) =====================
tt = sol.t
m_series = X[6, :]
dt = np.diff(tt)

# Normas de r e v para métricas de velocidade
r_norm_series = np.linalg.norm(X[0:3, :].T, axis=1)
v_norm_series = np.linalg.norm(X[3:6, :].T, axis=1)

# Máscaras
fire_mask = np.array([in_any_window(nu) for nu in nus_deg], dtype=bool)  # H só em janelas
u_mask    = (m_series > m_dry)                                           # tem propelente

# Acelerações instantâneas (km/s^2)
aV_series = (T_V / np.maximum(m_series, 1e-18)) / 1000.0
aH_series = (T_H / np.maximum(m_series, 1e-18)) / 1000.0

# Δv (km/s) → integrações
dv_V_kms = float(np.sum(aV_series[:-1] * dt * u_mask[:-1]))
dv_H_kms = float(np.sum(aH_series[:-1] * dt * (fire_mask[:-1] & u_mask[:-1])))

# Conversões p/ m/s
dv_V_ms = 1000.0 * dv_V_kms
dv_H_ms = 1000.0 * dv_H_kms
dv_total_ms = dv_V_ms + dv_H_ms

# tempo com H ligado (s)
t_H_on = float(np.sum(dt * (fire_mask[:-1])))

# -------- v no perigeu/apogeu --------
def peri_apo_speeds_from_elements(a: float, e: float, mu: float) -> tuple[float, float]:
    a = float(a); e = float(e)
    vp = np.sqrt(mu * (1.0 + e) / (a * (1.0 - e + 1e-32)))
    va = np.sqrt(mu * (1.0 - e) / (a * (1.0 + e + 1e-32)))
    return vp, va

def ang_wrap_deg(x):
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0

def mask_near_angle(nu_deg: np.ndarray, center_deg: float, half_width_deg: float) -> np.ndarray:
    return np.abs(ang_wrap_deg(nu_deg - center_deg)) <= half_width_deg

a_series = np.array([el.major_axis for el in orbital_elementss], dtype=float)
e_series = np.array([el.eccentricity for el in orbital_elementss], dtype=float)

vp0, va0 = peri_apo_speeds_from_elements(a_series[0],  e_series[0],  mu)
vpF, vaF = peri_apo_speeds_from_elements(a_series[-1], e_series[-1], mu)

PERI_HALF_WIDTH = 1.0
APO_HALF_WIDTH  = 1.0
peri_mask = mask_near_angle(nus_deg, 0.0,   PERI_HALF_WIDTH)
apo_mask  = mask_near_angle(nus_deg, 180.0, APO_HALF_WIDTH)

if np.any(peri_mask):
    v_peri_meas = float(np.mean(v_norm_series[peri_mask]))
else:
    v_peri_meas = float(np.max(v_norm_series))

if np.any(apo_mask):
    v_apo_meas = float(np.mean(v_norm_series[apo_mask]))
else:
    v_apo_meas = float(np.min(v_norm_series))

# ---------- Relatórios ----------
# Usa v_apogeu se a janela é centrada em 180°, senão v_perigeu
v_ref = v_apo_meas if 180.0 in MEAN_THETA_LIST_DEG else v_peri_meas
arg = np.clip((dv_H_kms) / (2.0 * v_ref), -1.0, 1.0)  # dv_H_kms está em km/s
delta_i_ideal_deg = float(np.degrees(2.0*np.arcsin(arg)))
delta_i_sim_deg = incs_deg - incs_deg[0]

print("\n=== Dados V H UP ===")
print(f"Tempo com H ligado (s):     {t_H_on:.6f}")
print(f"Δv_V     (m/s):             {dv_V_ms:.6f}")
print(f"Δv_H     (m/s):             {dv_H_ms:.6f}")
print(f"Δv_total (m/s):             {dv_total_ms:.6f}")
print(f"Δi_ideal (graus):           {delta_i_ideal_deg:.9f}")
print(f"Δi_sim (último - inicial):  {delta_i_sim_deg[-1]:.9f}")

if 0.0 in MEAN_THETA_LIST_DEG:
    print(f"→ Janela centrada no PERIGEU: usar v ≈ {v_peri_meas:.9f} km/s (medido) "
          f"ou {vpF:.9f} km/s (teórico no fim).")
if 180.0 in MEAN_THETA_LIST_DEG:
    print(f"→ Janela centrada no APOGEU: usar v ≈ {v_apo_meas:.9f} km/s (medido) "
          f"ou {vaF:.9f} km/s (teórico no fim).")

# ---------- plot dos elementos ----------
plot_classic_orbital_elements(t, orbital_elementss)

# 1) máscara "empuxo H ON"
thr_H_on_mask = (m_series > m_dry) & fire_mask

# 2) sombrear no gráfico
def _mask_to_spans(tarr, mask_bool):
    tarr = np.asarray(tarr, float); mask = np.asarray(mask_bool, bool)
    if mask.size == 0: return []
    rises  = np.where(np.diff(mask.astype(np.int8)) == +1)[0] + 1
    falls  = np.where(np.diff(mask.astype(np.int8)) == -1)[0] + 1
    if mask[0]:  rises = np.r_[0, rises]
    if mask[-1]: falls = np.r_[falls, mask.size-1]
    return list(zip(tarr[rises], tarr[falls]))

def add_thrust_spans(ax, tarr, mask_bool, *, color="tab:orange", alpha=0.15, label="Empuxo H ON"):
    for t0, t1 in _mask_to_spans(tarr, mask_bool):
        ax.axvspan(t0, t1, color=color, alpha=alpha, lw=0)
    handles, labels = ax.get_legend_handles_labels()
    patch = mpatches.Patch(color=color, alpha=alpha, label=label)
    if label not in labels:
        ax.legend(handles + [patch], labels + [label])

def _unwrap_deg(a_deg):
    return np.degrees(np.unwrap(np.radians(np.asarray(a_deg, float))))

Omega_series = [el.ascending_node for el in orbital_elementss]
Omega_unw = _unwrap_deg(Omega_series)

t_plot = t/86400.0  # dias
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
ax.plot3D(X[0, :], X[1, :], X[2, :], 'r', label="Satélite V_H UP")
ax.set_box_aspect([1, 1, 1])
ax.set_title("Órbita simulada - Satélite V_H UP")
ax.legend()
ax.axis('equal')

# ---------- i(ν) sem dente de serra ----------
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
plot_i_vs_nu_segmentado(nus_deg, incs_deg, ax=ax2, color="red", lw=1.5, label="i vs. ν")
ax2.legend()
plt.show()

def simulate():
    """
    Executa a integração com o x_dot deste módulo (VH UP) e devolve:
    (t, X, nus_deg, incs_deg, orbital_elementss)
    """
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
        nus_deg.append(nu)
        incs_deg.append(get_inclination(r_vec, v_vec, mu))

    return sol.t, X, np.array(nus_deg, float), np.array(incs_deg, float), orbital_elementss
