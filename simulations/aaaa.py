import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from utils.visualization import ElementsSeries, plot_classic_orbital_elements
from utils.orbital_elements import*
from utils.orbitalElementsOperations import*
import os

# ===================== condições iniciais =====================
r = np.array([6877.452, 0.0, 0.0])     # ITASAT-2 (circular ~LEO)
v = np.array([0.0, 5.383, 5.383])

t = np.linspace(0, 43200, 10000)  # 12 h
earth_radius = 6378.0  # km
mu = 3.986e5           # km^3/s^2
thrust = 0     # N

# ===================== Propulsão / massa =====================
T   = thrust    # N
Isp = 2150.0    # s
g0  = 9.80665   # m/s^2
m_sat = 20.0    # kg
m0    = 20.0    # kg
m_dry = 15.0    # kg

# ===================== Achatamento (J2 e J22 do ar_prs) =====================
from achatamento import EarthShapeParams as ShapeParams, accel_achatamento_total
_USE_J2  = False
_USE_J22 = False
_GAMMA   = 7.2921150e-5
LAMBDA22_DEG = -14.79
_SHAPE   = ShapeParams()
LAMBDA22_RAD = np.deg2rad(LAMBDA22_DEG)

def _lambdat_rad(tval: float) -> float:
    return _GAMMA * float(tval)

def _accel_achatamento(r_vec: np.ndarray, tval: float) -> np.ndarray:
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

# ===================== Janelas empuxo =====================
THRUST_INTERVAL_DEG = 30.0
MEAN_THETA_LIST_DEG = [180]

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

# ===================== Helpers robustos =====================
_EPS_E = 1e-5
_EPS_I = np.deg2rad(1e-3)

def _safe_norm(x):
    n = np.linalg.norm(x)
    return n if n > 1e-32 else 1e-32

def _argument_of_latitude_deg(r_vec, v_vec):
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
    r_vec = x[0:3]
    v_vec = x[3:6]
    rnorm = np.linalg.norm(r_vec) + 1e-32
    xdot[3:6] = -(mu/(rnorm**3))*r_vec
    xdot[3:6] += _accel_achatamento(r_vec, t)
    m_cur = max(x[6], 1e-18)
    xdot[3:6] += _accel_DRAG(r_vec, v_vec, m_cur)
    u = throttle(t, x)
    a_inst = (T / m_cur) / 1000.0
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

# ===================== Integração =====================
x0 = np.concatenate((r, v, [m0]))
sol = solve_ivp(x_dot, (t[0], t[-1]), x0, t_eval=t, method="DOP853", rtol=1e-12, atol=1e-15)
X = sol.y

# ===================== Plot dos elementos =====================
a_series = np.array([get_major_axis(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])])
e_series = np.array([get_eccentricity(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])])
i_deg_series = np.array([get_inclination(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])])
Om_deg_series = np.array([get_ascending_node(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])])
w_deg_series = np.array([get_argument_of_perigee(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])])
nu_deg_series = np.array([get_true_anomaly(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])])
u_deg_series = np.array([get_argument_of_latitude(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])])
ltrue_deg_series = np.array([get_true_longitude(X[0:3, k], X[3:6, k], mu) for k in range(X.shape[1])])
energy_series = np.array([0.5*np.dot(X[3:6, k], X[3:6, k]) - mu/np.linalg.norm(X[0:3, k]) for k in range(X.shape[1])])
elems = ElementsSeries(a_series, e_series, i_deg_series, Om_deg_series, w_deg_series, nu_deg_series, u_deg_series, ltrue_deg_series, energy_series)

plot_classic_orbital_elements(t, elems)

# ===================== Função para Terra texturizada =====================
def draw_textured_earth(ax, radius_km: float, texture_path: str,
                        n_long: int = 360, n_lat: int = 180, alpha: float = 1.0):
    if not os.path.isfile(texture_path):
        raise FileNotFoundError(f"Textura não encontrada: {texture_path}")

    img = plt.imread(texture_path)
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0

    H, W = img.shape[:2]
    u = np.linspace(0, 2*np.pi, n_long)
    v = np.linspace(0, np.pi, n_lat)
    uu, vv = np.meshgrid(u, v)
    x = radius_km * np.cos(uu) * np.sin(vv)
    y = radius_km * np.sin(uu) * np.sin(vv)
    z = radius_km * np.cos(vv)
    iu = (uu / (2*np.pi) * (W - 1)).astype(int)
    iv = (vv / np.pi * (H - 1)).astype(int)
    iv = (H - 1) - iv
    rgb = img[iv, iu, :3]
    a = np.full((*rgb.shape[:2], 1), float(alpha), dtype=rgb.dtype)
    facecolors = np.dstack([rgb, a])
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=facecolors,
                    linewidth=0, antialiased=False, shade=False)

# ============ Função utilitária: eixos iguais (evita Terra "oval") ============
def set_axes_equal(ax):
    """Força as escalas X=Y=Z em um Axes3D."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max([x_range, y_range, z_range]) / 2.0

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([x_mid - max_range, x_mid + max_range])
    ax.set_ylim3d([y_mid - max_range, y_mid + max_range])
    ax.set_zlim3d([z_mid - max_range, z_mid + max_range])

# ===================== Plot 3D com textura (com eixos e grade) =====================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

TEXTURE_PATH = r"c:\Users\ppggo\Documents\GitHub\a\simulations\earth.jpg"
draw_textured_earth(ax, radius_km=earth_radius, texture_path=TEXTURE_PATH)

# Órbita
ax.plot3D(X[0, :], X[1, :], X[2, :], color='lime', lw=2.0, label="Órbita (V-only)")

# Labels, título e legenda
ax.set_title("Órbita simulada - Terra Texturizada", pad=14)
ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_zlabel("Z [km]")
ax.legend(loc="upper right")

# Limites simétricos baseados na maior distância + margem
rmax = float(np.max(np.linalg.norm(X[0:3, :], axis=0)))
R = max(earth_radius, rmax) * 1.15
ax.set_xlim(-R, R)
ax.set_ylim(-R, R)
ax.set_zlim(-R, R)

# Aspecto cúbico e escalas iguais (sem “oval”)
ax.set_box_aspect([1, 1, 1])
set_axes_equal(ax)

# Ticks e grade
ticks = np.linspace(-R, R, 5)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)
ax.grid(True)

plt.tight_layout()
plt.show()
