import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from utils.visualization import ElementsSeries, plot_classic_orbital_elements
from utils.orbital_elements import*
from utils.orbitalElementsOperations import*

# ===================== condições iniciais =====================
# Vetores de estado (ECI, km e km/s)
r = np.array([6877.452, 0.0, 0.0])     # ITASAT-2 (circular ~LEO)
v = np.array([0.0, 5.383, 5.383])

# r = np.array([2128.5832, -6551.1055, 0.0000])   # ITASAT-1 com nodo ascendente em 280°
# v = np.array([-0.932454, -0.302973, 7.548972])
# r = np.array([10016.34, -17012.52, 7899.28])    # Exemplo elíptico
# v = np.array([2.5, -1.05, 3.88])

t = np.linspace(0, 43200, 10000)  # 12 h
earth_radius = 6378.0  # km
mu = 3.986e5           # km^3/s^2
thrust = 1.1e-3        # N

# ===================== Propulsão / massa =====================
T   = thrust    # N
Isp = 2150.0    # s
g0  = 9.80665   # m/s^2
m_sat = 20.0    # kg
m0    = 20.0    # kg
m_dry = 15.0    # kg

# ===================== Achatamento (J2 e J22 do ar_prs) =====================
from achatamento import (
    EarthShapeParams as ShapeParams,
    accel_achatamento_total,
)
_USE_J2  = True
_USE_J22 = False          # ligue para testar
_GAMMA   = 7.2921150e-5     # rad/s, rotação da Terra em ECI (tesseral “gira”)
LAMBDA22_DEG = -14.79 # lambdat = gamma * t (rad) 
_SHAPE   = ShapeParams()  # μ, Re, J2, J22

# Ângulos do termo tesseral J22 (iguais ao .for)
LAMBDA22_RAD = np.deg2rad(LAMBDA22_DEG)

def _lambdat_rad(tval: float) -> float:
    return _GAMMA * float(tval)

def _accel_achatamento(r_vec: np.ndarray, tval: float) -> np.ndarray:
    """
    Aceleração de achatamento (J2 + opcional J22) conforme ar_prs.
    r_vec em km, saída em km/s^2, frame equatorial inercial (z || spin da Terra).
    """
    return accel_achatamento_total(
        r_vec, _SHAPE,
        lambdat_rad=_lambdat_rad(tval),
        lambda22_rad=LAMBDA22_RAD,
        use_j2=_USE_J2,
        use_j22=_USE_J22
    )

# ===================== Arrasto atmosférico (opcional) =====================
from Drag import accel_drag, DragParams
_DRAG_ON = False
_DRAG = DragParams(Cd=2.2, A_ref_m2=0.02, use_atmo_rotation=True,
                   rho0_kg_m3=3.614e-11, h0_km=200.0, H_km=50.0)

def _accel_DRAG(r_vec, v_vec, m_cur):
    return accel_drag(r_vec, v_vec, m_cur, _DRAG) if _DRAG_ON else 0.0 * r_vec

# ===================== Janelas empuxo (não usadas aqui, mas mantidas) =====================
THRUST_INTERVAL_DEG = 30.0
MEAN_THETA_LIST_DEG = [180]  # apogeu

def throttle(t, x):
    if T <= 0.0:
        return 0.0
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

# ===================== Helpers robustos (circular/equatorial) =====================
_EPS_E = 1e-5
_EPS_I = np.deg2rad(1e-3)

def _safe_norm(x):
    n = np.linalg.norm(x)
    return n if n > 1e-32 else 1e-32

def _argument_of_latitude_deg(r_vec, v_vec):
    """
    u = argumento da latitude, robusto quando e ~ 0 e i não ~ 0.
    Se equatorial (i ~ 0), usa longitude verdadeira l_true.
    """
    r = np.array(r_vec, float).reshape(3)
    v = np.array(v_vec, float).reshape(3)
    rnorm = _safe_norm(r)
    h = np.cross(r, v)
    hnorm = _safe_norm(h)
    k = np.array([0.0, 0.0, 1.0])
    n = np.cross(k, h)
    nnorm = np.linalg.norm(n)

    i_rad = np.arccos(np.clip(h[2]/hnorm, -1.0, 1.0))
    if i_rad <= _EPS_I or nnorm < 1e-14:
        return np.degrees(np.arctan2(r[1], r[0])) % 360.0

    cosu = np.dot(n, r)/(nnorm*rnorm)
    sinu = np.dot(np.cross(n, r), h)/(nnorm*rnorm*hnorm)
    u = np.degrees(np.arctan2(sinu, np.clip(cosu, -1.0, 1.0))) % 360.0
    return u

# ===================== Dinâmica =====================
def x_dot(t, x):
    xdot = np.zeros_like(x)
    xdot[0:3] = x[3:6]

    # Gravidade central
    r_vec = x[0:3]
    v_vec = x[3:6]
    rnorm = np.linalg.norm(r_vec) + 1e-32
    xdot[3:6] = -(mu/(rnorm**3))*r_vec

    # Achatamento (J2 + opcional J22)
    xdot[3:6] += _accel_achatamento(r_vec, t)

    # Arrasto (opcional)
    m_cur = max(x[6], 1e-18)
    xdot[3:6] += _accel_DRAG(r_vec, v_vec, m_cur)

    # Propulsão V-only (along-track)
    u = throttle(t, x)
    a_inst = (T / m_cur) / 1000.0  # km/s^2

    # Base RSW
    r_hat = r_vec / rnorm
    h_vec_local = np.cross(r_vec, v_vec)
    w_hat = h_vec_local / (_safe_norm(h_vec_local))
    s_hat = np.cross(w_hat, r_hat)
    s_hat /= _safe_norm(s_hat)

    if u > 0.0:
        xdot[3:6] += a_inst * s_hat
        xdot[6] = - T/(Isp*g0)
    else:
        xdot[6] = 0.0

    return xdot

# >>> Estado inicial inclui massa <<<
x0 = np.concatenate((r, v, [m0]))

sol = solve_ivp(
    x_dot, (t[0], t[-1]), x0, t_eval=t,
    method="DOP853",
    rtol=1e-12, atol=1e-15,
)
X = sol.y

# ---------- elementos, ν/u e i ----------
orbital_elementss = []
nus_deg = []
incs_deg = []

for k in range(X.shape[1]):
    r_vec = X[0:3, k]
    v_vec = X[3:6, k]

    el = get_orbital_elements(r_vec, v_vec, mu)
    orbital_elementss.append(el)

    e_now = get_eccentricity(r_vec, v_vec, mu)
    if e_now < _EPS_E:
        nu_or_u = _argument_of_latitude_deg(r_vec, v_vec)
    else:
        nu_or_u = get_true_anomaly(r_vec, v_vec, mu)
    nus_deg.append(nu_or_u)

    incs_deg.append(get_inclination(r_vec, v_vec, mu))

nus_deg = np.array(nus_deg, dtype=float)
incs_deg = np.array(incs_deg, dtype=float)

# ---------- Δv numérico ----------
tt = sol.t
m_series = X[6, :]
dt = np.diff(tt)
aV_series = (T / np.maximum(m_series, 1e-18)) / 1000.0  # km/s^2
u_mask = (m_series > m_dry)

dv_V_kms = float(np.sum(aV_series[:-1] * dt * u_mask[:-1]))
dv_V_ms  = 1000.0 * dv_V_kms
dv_H_ms  = 0.0
dv_total_ms = dv_V_ms + dv_H_ms

print("\n=== Δv acumulado (V-only) ===")
print(f"Δv_V     (m/s): {dv_V_ms:.6f}")
print(f"Δv_H     (m/s): {dv_H_ms:.6f}")
print(f"Δv_total (m/s): {dv_total_ms:.6f}")

# ---------- utilitário: plot i(nu) segmentado ----------
def plot_i_vs_nu_segmentado(nu_deg: np.ndarray, inc_deg: np.ndarray, *, ax=None,
                            color="black", **plot_kw):
    nu_deg = np.asarray(nu_deg, dtype=float)
    inc_deg = np.asarray(inc_deg, dtype=float)
    if ax is None:
        fig, ax = plt.subplots()
    dn = np.diff(nu_deg)
    wraps = np.where(dn < -180.0)[0]
    start = 0
    for w in wraps:
        ax.plot(nu_deg[start:w+1], inc_deg[start:w+1], color=color, **plot_kw)
        start = w + 1
    ax.plot(nu_deg[start:], inc_deg[start:], color=color, **plot_kw)
    ax.set_xlabel(r'$\nu$ ou $u$ (deg)')
    ax.set_ylabel(r'$i$ (deg)')
    ax.set_xlim(0.0, 360.0)
    ax.grid(True)
    return ax

# ---------- construir ElementsSeries ----------
a_series      = np.array([get_major_axis      (X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], dtype=float)
e_series      = np.array([get_eccentricity    (X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], dtype=float)
i_deg_series  = np.array([get_inclination     (X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], dtype=float)
Om_deg_series = np.array([get_ascending_node  (X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], dtype=float)
w_deg_series  = np.array([get_argument_of_perigee(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], dtype=float)
nu_deg_series = np.array([get_true_anomaly    (X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], dtype=float)

u_deg_series     = np.array([get_argument_of_latitude(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], dtype=float)
ltrue_deg_series = np.array([get_true_longitude     (X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], dtype=float)

try:
    energy_series = np.array([get_specific_energy(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], dtype=float)
except NameError:
    def _specific_energy(r, v, mu):
        return 0.5*np.dot(v, v) - mu/np.linalg.norm(r)
    energy_series = np.array([_specific_energy(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])], dtype=float)

elems = ElementsSeries(
    a=a_series,
    e=e_series,
    i_deg=i_deg_series,
    Omega_deg=Om_deg_series,
    omega_deg=w_deg_series,
    nu_deg=nu_deg_series,
    u_deg=u_deg_series,
    ltrue_deg=ltrue_deg_series,
    energy=energy_series
)

# ---------- plot dos elementos ----------
plot_classic_orbital_elements(t, elems)

# ---------- plot 3D ----------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
u_grid, vgrid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = earth_radius * np.cos(u_grid) * np.sin(vgrid)
y_e = earth_radius * np.sin(u_grid) * np.sin(vgrid)
z_e = earth_radius * np.cos(vgrid)
ax.plot_wireframe(x_e, y_e, z_e, color="g", alpha=0.3)
ax.plot3D(X[0, :], X[1, :], X[2, :], 'g', label="Satélite V-only")
ax.set_box_aspect([1, 1, 1])
ax.set_title("Órbita simulada - Satélite V-only")
ax.legend()
ax.axis('equal')

# ---------- plot i(nu/u) ----------
fig2, ax2 = plt.subplots()
plot_i_vs_nu_segmentado(nus_deg, incs_deg, ax=ax2, color="green", lw=1.5, label='i vs. (ν ou u)')
ax2.legend()
plt.show()

# ===================== Interface de simulação =====================
def simulate():
    """
    Executa a integração com o x_dot deste módulo e devolve:
    (t, X, nus_deg, incs_deg, orbital_elementss)
    onde:
      - t: vetor de tempo (s)
      - X: estados [x,y,z,vx,vy,vz,m] ao longo do tempo (7 x N)
      - nus_deg: ângulo de fase (ν ou u) em graus, [0, 360)
      - incs_deg: inclinação em graus
      - orbital_elementss: lista de OrbitalElements por amostra
    """
    x0 = np.concatenate((r, v, [m0]))
    sol = solve_ivp(
        x_dot, (t[0], t[-1]), x0, t_eval=t,
        method="DOP853", rtol=1e-12, atol=1e-15,
    )
    X = sol.y

    orbital_elementss = []
    nus_deg = []
    incs_deg = []

    for k in range(X.shape[1]):
        r_vec = X[0:3, k]
        v_vec = X[3:6, k]

        el = get_orbital_elements(r_vec, v_vec, mu)
        orbital_elementss.append(el)

        e_now = get_eccentricity(r_vec, v_vec, mu)
        if e_now < _EPS_E:
            nu_or_u = _argument_of_latitude_deg(r_vec, v_vec)
        else:
            nu_or_u = get_true_anomaly(r_vec, v_vec, mu)

        nus_deg.append(nu_or_u)
        incs_deg.append(get_inclination(r_vec, v_vec, mu))

    return sol.t, X, np.array(nus_deg, float), np.array(incs_deg, float), orbital_elementss
from utils.eccentricity import plot_eccentricity_time, plot_eccentricity_in_orbital_plane

# 1) componentes e módulo vs. tempo
plot_eccentricity_time(t, X, mu)
plt.show()

# 2) trajetória de e no plano orbital (mostra rotação do perigeu)
plot_eccentricity_in_orbital_plane(X, mu)
plt.show()
