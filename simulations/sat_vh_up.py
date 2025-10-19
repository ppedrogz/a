import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.visualization import ElementsSeries, plot_classic_orbital_elements
from utils.orbital_elements import*
from utils.orbitalElementsOperations import*

# ===================== condições iniciais =====================
r = np.array([6877.452, 0.0, 0.0])  # parametros orbitais ITASAT-2 (LEO quase circular)
v = np.array([0.0, 5.383, 5.383])

t = np.linspace(0, 43200, 10000)  # 12 h
earth_radius = 6378.0  # km
mu = 3.986e5           # km^3/s^2

# ===================== Dados de propulsão =====================
T   = 1.1e-3       # N
Isp = 2150.0       # s
g0  = 9.80665      # m/s^2
m0  = 20.0         # kg (inicial)
m_dry = 15.0       # kg (massa seca)

# =============================================================================
#  ACHATAMENTO (J2 + J22) compatível com o ar_prs.for
# =============================================================================
from achatamento import (
    EarthShapeParams as ShapeParams,
    accel_achatamento_total,
)

# ---- flags globais (podem ser sobrescritos por simulate(...)) ----
_USE_J2  = False    # deixe True para incluir J2
_USE_J22 = False    # ligue para incluir J22 tesseral
# rotação sideral da Terra (rad/s) — o padrão tesseral “gira” no ECI
_GAMMA         = 7.2921150e-5
# longitude do eixo do termo J22 (modelo GEM/ar_prs): -14.79 graus
LAMBDA22_DEG   = -14.79
LAMBDA22_RAD   = np.deg2rad(LAMBDA22_DEG)
_SHAPE         = ShapeParams()  # μ, Re, J2, J22

def _lambdat_rad(tval: float) -> float:
    """Ângulo de rotação da Terra no ECI (rad) para o termo tesseral."""
    return _GAMMA * float(tval)

def _accel_achatamento(r_vec: np.ndarray, tval: float) -> np.ndarray:
    """
    Aceleração de achatamento (J2 + opcional J22), em km/s^2,
    frame inercial equatorial (z || eixo de rotação da Terra).
    """
    return accel_achatamento_total(
        r_vec, _SHAPE,
        lambdat_rad=_lambdat_rad(tval),
        lambda22_rad=LAMBDA22_RAD,
        use_j2=_USE_J2,
        use_j22=_USE_J22,
    )

# -----------------------------------------------------------------------------
# [ANTIGO] J2 “genérico” — mantido como referência (comentado)
# -----------------------------------------------------------------------------
# from J2 import external_accel, EarthParams, PerturbationFlags
# _J2_ON = False
# def _accel_J2(r_vec, v_vec, tval):
#     if _J2_ON:
#         return external_accel(r_vec, v_vec, tval,
#                               params=EarthParams(),
#                               flags=PerturbationFlags(j2=True))
#     return 0.0 * r_vec

# ===================== Arrasto (opcional) =====================
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

# ===================== Janelas em ângulo orbital =====================
THRUST_INTERVAL_DEG = 30.0
MEAN_THETA_LIST_DEG = [180]  # 180 = apogeu; 0 = perigeu

def wrap_deg(a): return np.remainder(a, 360.0)

def angle_in_window_deg(theta_deg, center_deg, width_deg):
    half = 0.5*width_deg
    lo = wrap_deg(center_deg - half)
    hi = wrap_deg(center_deg + half)
    th = wrap_deg(theta_deg)
    return (th >= lo) & (th <= hi) if lo <= hi else ((th >= lo) | (th <= hi))

def in_any_window(theta_deg):
    return any(angle_in_window_deg(theta_deg, cdeg, THRUST_INTERVAL_DEG)
               for cdeg in MEAN_THETA_LIST_DEG)

def throttle(tval, x):
    if T <= 0.0: return 0.0
    return 1.0 if x[6] > m_dry else 0.0

# ===================== Robustez p/ circular/equatorial =====================
_EPS_E = 1e-5
_EPS_I = np.deg2rad(1e-3)

def _safe_norm(x):
    n = np.linalg.norm(x);  return n if n > 1e-32 else 1e-32

def phase_angle_deg(r_vec, v_vec, mu):
    """ν (anomalia verdadeira) ou u (arg. de latitude) se e≈0 (e l_true se i≈0)."""
    e_now = get_eccentricity(r_vec, v_vec, mu)
    return get_argument_of_latitude(r_vec, v_vec, mu) if e_now < _EPS_E else get_true_anomaly(r_vec, v_vec, mu)

# ===================== Dinâmica (VH UP) =====================
def x_dot(tval, x):
    xdot = np.zeros_like(x)
    xdot[0:3] = x[3:6]

    r_vec = x[0:3];  v_vec = x[3:6]
    rnorm = _safe_norm(r_vec)

    # Gravidade 2-corpos
    xdot[3:6] = -(mu/(rnorm**3))*r_vec

    # ======= ACHATAMENTO (J2 + opcional J22) =======
    xdot[3:6] += _accel_achatamento(r_vec, tval)

    # Arrasto (se ligado)
    m_cur = max(x[6], 1e-18)
    xdot[3:6] += _accel_DRAG(r_vec, v_vec, m_cur)

    # Base RSW
    r_hat = r_vec / rnorm
    h_vec_local = np.cross(r_vec, v_vec)
    w_hat = h_vec_local / _safe_norm(h_vec_local)
    s_hat = np.cross(w_hat, r_hat); s_hat /= _safe_norm(s_hat)

    # Ângulo de fase (ν ou u)
    theta_deg = phase_angle_deg(r_vec, v_vec, mu)
    fire_H = in_any_window(theta_deg)

    # Empuxo
    u_thr = throttle(tval, x)

    if not DUAL_THRUSTERS:
        a_inst = (T / m_cur) / 1000.0  # km/s^2
        alpha = np.deg2rad(45.0) if fire_H else 0.0
        dir_hat = np.cos(alpha)*s_hat + np.sin(alpha)*w_hat
        if u_thr > 0.0:
            xdot[3:6] += a_inst * dir_hat
            xdot[6] = - T/(Isp*g0)
        else:
            xdot[6] = 0.0
    else:
        aV = (T_V / m_cur) / 1000.0
        aH = (T_H / m_cur) / 1000.0 if fire_H else 0.0
        # Sinal DOWN: alterna por u (ou l_true se i≈0)
        u_deg = get_argument_of_latitude(r_vec, v_vec, mu)
        SIGN_H = np.sign(np.cos(np.deg2rad(u_deg)))
        if u_thr > 0.0:
            xdot[3:6] += aV * s_hat + SIGN_H * aH * w_hat
            mdot_V = T_V/(Isp_V*g0)
            mdot_H = (T_H/(Isp_H*g0)) if fire_H else 0.0
            xdot[6] = - (mdot_V + mdot_H)
        else:
            xdot[6] = 0.0

    return xdot

# >>> Integração (estado inclui massa) <<<
x0 = np.concatenate((r, v, [m0]))
sol = solve_ivp(
    x_dot, (t[0], t[-1]), x0, t_eval=t,
    method="DOP853",
    rtol=1e-13, atol=1e-17,
)
X = sol.y

# ----------------- elementos para plots -----------------
a_series      = np.array([get_major_axis         (X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], float)
e_series      = np.array([get_eccentricity       (X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], float)
i_deg_series  = np.array([get_inclination        (X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], float)
Om_deg_series = np.array([get_ascending_node     (X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], float)
w_deg_series  = np.array([get_argument_of_perigee(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], float)
nu_deg_series = np.array([get_true_anomaly      (X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], float)
u_deg_series     = np.array([get_argument_of_latitude(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], float)
ltrue_deg_series = np.array([get_true_longitude     (X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], float)

# Energia específica
try:
    energy_series = np.array([get_specific_energy(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], float)
except Exception:
    def _specific_energy(r, v, mu): return 0.5*np.dot(v, v) - mu/np.linalg.norm(r)
    energy_series = np.array([_specific_energy(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], float)

# ---------- elementos, fase e inclinação (para i vs fase) ----------
orbital_elementss = []
phase_deg = []; incs_deg = []
for k in range(X.shape[1]):
    r_vec = X[0:3, k]; v_vec = X[3:6, k]
    orbital_elementss.append(get_orbital_elements(r_vec, v_vec, mu))
    phase_deg.append(phase_angle_deg(r_vec, v_vec, mu))
    incs_deg.append(get_inclination(r_vec, v_vec, mu))
phase_deg = np.array(phase_deg, float)
incs_deg  = np.array(incs_deg,  float)

# ===================== MÉTRICAS (Δv, janelas, etc.) =====================
tt = sol.t
m_series = X[6, :]
dt = np.diff(tt)

r_norm_series = np.linalg.norm(X[0:3, :].T, axis=1)
v_norm_series = np.linalg.norm(X[3:6, :].T, axis=1)
fire_mask = np.array([in_any_window(th) for th in phase_deg], bool)
u_mask    = (m_series > m_dry)

if DUAL_THRUSTERS:
    aV_series = (T_V / np.maximum(m_series, 1e-18)) / 1000.0
    aH_series = (T_H / np.maximum(m_series, 1e-18)) / 1000.0
else:
    aV_series = (T   / np.maximum(m_series, 1e-18)) / 1000.0
    aH_series = 0.0 * aV_series

dv_V_kms = float(np.sum(aV_series[:-1] * dt * u_mask[:-1]))
dv_H_kms = float(np.sum(aH_series[:-1] * dt * (fire_mask[:-1] & u_mask[:-1])))
dv_V_ms, dv_H_ms = 1000.0*dv_V_kms, 1000.0*dv_H_kms
dv_total_ms = dv_V_ms + dv_H_ms

t_H_on = float(np.sum(dt * (fire_mask[:-1])))

def peri_apo_speeds_from_elements(a: float, e: float, mu: float):
    a = float(a); e = float(e)
    vp = np.sqrt(mu * (1.0 + e) / (a * (1.0 - e + 1e-32)))
    va = np.sqrt(mu * (1.0 - e) / (a * (1.0 + e + 1e-32)))
    return vp, va

def ang_wrap_deg(x): return (np.asarray(x, float) + 180.0) % 360.0 - 180.0
def mask_near_angle(phi_deg, center_deg, half_width_deg): return np.abs(ang_wrap_deg(phi_deg - center_deg)) <= half_width_deg

vp0, va0 = peri_apo_speeds_from_elements(a_series[0],  e_series[0],  mu)
vpF, vaF = peri_apo_speeds_from_elements(a_series[-1], e_series[-1], mu)

PERI_HALF_WIDTH = 1.0;  APO_HALF_WIDTH = 1.0
peri_mask = mask_near_angle(phase_deg, 0.0,   PERI_HALF_WIDTH)
apo_mask  = mask_near_angle(phase_deg, 180.0, APO_HALF_WIDTH)
v_peri_meas = float(np.mean(v_norm_series[peri_mask])) if np.any(peri_mask) else float(np.max(v_norm_series))
v_apo_meas  = float(np.mean(v_norm_series[apo_mask]))  if np.any(apo_mask)  else float(np.min(v_norm_series))

v_ref = v_apo_meas if 180.0 in MEAN_THETA_LIST_DEG else v_peri_meas
arg = np.clip((dv_H_kms) / (2.0 * v_ref), -1.0, 1.0)
delta_i_ideal_deg = float(np.degrees(2.0*np.arcsin(arg)))
delta_i_sim_deg   = incs_deg - incs_deg[0]

print("\n=== Dados V_H UP ===")
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
elems = ElementsSeries(
    a=a_series, e=e_series, i_deg=i_deg_series,
    Omega_deg=Om_deg_series, omega_deg=w_deg_series,
    nu_deg=nu_deg_series, u_deg=u_deg_series,
    ltrue_deg=ltrue_deg_series, energy=energy_series
)
plot_classic_orbital_elements(t, elems)

#  Sombras de empuxo H nas janelas
thr_H_on_mask = (m_series > m_dry) & fire_mask
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

def _unwrap_deg(a_deg): return np.degrees(np.unwrap(np.radians(np.asarray(a_deg, float))))

Omega_series = Om_deg_series
Omega_unw = _unwrap_deg(Omega_series)

t_plot = t/86400.0  # dias
fig, ax = plt.subplots()
ax.plot(t_plot, Omega_unw, 'r-', lw=1.2, label='Ω (desenrolado)')
add_thrust_spans(ax, t_plot, thr_H_on_mask, color="tab:orange", alpha=0.18, label="Empuxo H ON")
ax.set_xlabel("Tempo [dias]"); ax.set_ylabel("Ω [graus]")
ax.set_title("RAAN (VH UP) com janelas de empuxo H destacadas")
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
ax.legend(); ax.axis('equal')

# ---------- i(ângulo de fase) sem dente de serra ----------
def plot_i_vs_phase_segmentado(phi_deg: np.ndarray, inc_deg: np.ndarray, *, ax=None, **plot_kw):
    phi_deg = np.asarray(phi_deg, float); inc_deg = np.asarray(inc_deg, float)
    if ax is None: fig_loc, ax = plt.subplots()
    dn = np.diff(phi_deg); wraps = np.where(dn < -180.0)[0]
    start = 0
    for w in wraps:
        ax.plot(phi_deg[start:w+1], inc_deg[start:w+1], **plot_kw); start = w + 1
    ax.plot(phi_deg[start:], inc_deg[start:], **plot_kw)
    ax.set_xlabel(r'Fase (ν ou $u$) [deg]'); ax.set_ylabel(r'$i$ [deg]')
    ax.set_xlim(0.0, 360.0); ax.grid(True); return ax

fig2, ax2 = plt.subplots()
plot_i_vs_phase_segmentado(phase_deg, incs_deg, ax=ax2, color="red", lw=1.5, label="i vs. fase")
ax2.legend(); plt.show()

# ===================== Interface com flags (j2, j22, drag) =====================
def simulate(j2: bool = True, j22: bool = False, drag: bool = False):
    """
    Executa a integração com a dinâmica deste módulo (V_H UP) e devolve:
      (t, X, nus_deg, incs_deg, orbital_elementss)
    Parâmetros:
      j2   : liga/desliga J2 (achatamento zonal)
      j22  : liga/desliga J22 (tesseral de ordem 2)
      drag : liga/desliga arrasto
    OBS: 'nus_deg' é a FASE robusta (ν ou u/l_true) em graus.
    """
    global _USE_J2, _USE_J22, _DRAG_ON
    _USE_J2, _USE_J22, _DRAG_ON = bool(j2), bool(j22), bool(drag)

    x0 = np.concatenate((r, v, [m0]))
    sol = solve_ivp(
        x_dot, (t[0], t[-1]), x0, t_eval=t,
        method="DOP853", rtol=1e-12, atol=1e-15,
    )
    X = sol.y

    orbital_elementss = []
    phi_deg = []; incs_deg = []

    for k in range(X.shape[1]):
        r_vec = X[0:3, k]; v_vec = X[3:6, k]
        orbital_elementss.append(get_orbital_elements(r_vec, v_vec, mu))
        phi_deg.append(phase_angle_deg(r_vec, v_vec, mu))
        incs_deg.append(get_inclination(r_vec, v_vec, mu))

    return sol.t, X, np.array(phi_deg, float), np.array(incs_deg, float), orbital_elementss
