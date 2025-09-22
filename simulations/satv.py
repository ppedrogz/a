import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from utils.GetClassicOrbitalElements import *
from utils.visualization import plot_classic_orbital_elements

# ===================== condições iniciais =====================
r = np.array([10016.34, -17012.52, 7899.28])
v = np.array([2.5, -1.05, 3.88])
t = np.linspace(0, 4320000, 1000000)  
earth_radius = 6378.0  # km
mu = 3.986e5
thrust = 1.1e-3  # N (força fixa)

# ===================== ADIÇÕES (massa e Busek BIT-3) =====================
T   = thrust     # N
Isp = 2150.0     # s
g0  = 9.80665    # m/s^2

m_sat = 20.0     # kg
m0    = 20.0     # kg
m_dry = 15.0     # kg
# ========================================================================

# ===== Controle de tolerâncias (precisão do integrador) =====
# Ex.: ERROR_PERCENT = 0.001 -> rtol = 1e-5 (0,001%)
ERROR_PERCENT = 0.001
RTOL = ERROR_PERCENT / 100.0
ATOL = np.array([
    1e-3, 1e-3, 1e-3,   # posição [km] (1e-3 km = 1 m)
    1e-6, 1e-6, 1e-6,   # velocidade [km/s] (1e-6 km/s = 1 mm/s)
    1e-8                # massa [kg]
], dtype=float)

# adição da perturbação (J2)
from J2 import external_accel, EarthParams, PerturbationFlags

_J2_ON = False # flag global para ligar/desligar J2
def _accel_J2(r_vec, v_vec, tcur):
    if _J2_ON:
        return external_accel(r_vec, v_vec, tcur,
                              params=EarthParams(),
                              flags=PerturbationFlags(j2=True))
    return 0.0 * r_vec  # vetor zero

# ===================== utilitários de rotação (inclinação alvo) =====================
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
    O  = np.deg2rad(raan_deg)
    w  = np.deg2rad(argp_deg)
    nu = np.deg2rad(nu_deg)
    p = a * (1.0 - e**2)
    r_pqw = np.array([p*np.cos(nu)/(1.0 + e*np.cos(nu)),
                      p*np.sin(nu)/(1.0 + e*np.cos(nu)),
                      0.0])
    v_pqw = np.array([-np.sqrt(mu/p)*np.sin(nu),
                       np.sqrt(mu/p)*(e + np.cos(nu)),
                       0.0])
    C = _R3(O) @ _R1(i) @ _R3(w)
    return C @ r_pqw, C @ v_pqw

def reseed_state_with_inclination(r, v, mu, i_target_deg):
    a0  = get_major_axis(r, v, mu)
    e0  = get_eccentricity(r, v, mu)
    O0  = get_ascending_node(r, v, mu)
    w0  = get_argument_of_perigee(r, v, mu)
    nu0 = get_true_anormaly(r, v, mu)
    return _coe_to_rv(a0, e0, i_target_deg, O0, w0, nu0, mu)

# ======== alvo de inclinação inicial (em graus) ========
I0_TARGET_DEG = 97.5
r, v = reseed_state_with_inclination(r, v, mu, I0_TARGET_DEG)
print(f"[check] i0 = {get_inclination(r, v, mu):.6f} deg")

# ===================== (janelas por nu — não usadas no V-only) =====================
THRUST_INTERVAL_DEG = 30.0
MEAN_THETA_LIST_DEG = [180.0]

def throttle(tcur, x):
    return 1.0 if x[6] > m_dry else 0.0

# ===================== Dinâmica =====================
def x_dot(tcur, x):
    dx = np.zeros_like(x)
    r_vec = x[0:3]; v_vec = x[3:6]
    rnorm = np.linalg.norm(r_vec) + 1e-32

    # cinemática
    dx[0:3] = v_vec

    # gravidade + J2
    dx[3:6] = -(mu/(rnorm**3))*r_vec
    dx[3:6] += _accel_J2(r_vec, v_vec, tcur)

    # thrust tangencial (V-only)
    m_cur = max(x[6], 1e-18)
    u = throttle(tcur, x)
    a_inst = (T / m_cur) / 1000.0  # km/s^2

    # base RSW
    r_hat = r_vec / rnorm
    h_vec = np.cross(r_vec, v_vec)
    w_hat = h_vec / (np.linalg.norm(h_vec) + 1e-32)
    s_hat = np.cross(w_hat, r_hat); s_hat /= (np.linalg.norm(s_hat) + 1e-32)

    if u > 0.0:
        dx[3:6] += a_inst * s_hat
        dx[6] = - T/(Isp*g0)   # kg/s
    else:
        dx[6] = 0.0
    return dx

# >>> estado inicial inclui massa <<<
x0 = np.concatenate((r, v, [m0]))

# ===================== Integração COM solução densa =====================
sol = solve_ivp(
    x_dot, (t[0], t[-1]), x0,
    method='RK45', rtol=RTOL, atol=ATOL,
    dense_output=True
)

# estados amostrados na SUA grade:
X = sol.sol(t)          # shape (7, len(t))
tt = t.copy()           # use sempre 'tt' daqui pra frente

# ---------- elementos, ν (0–360) e i ----------
orbital_elementss = []
nus_deg = []
incs_deg = []
for k in range(X.shape[1]):
    r_vec = X[0:3, k]
    v_vec = X[3:6, k]
    orbital_elementss.append(get_orbital_elements(r_vec, v_vec, mu))
    nus_deg.append(get_true_anormaly(r_vec, v_vec, mu))  # já [0,360)
    incs_deg.append(get_inclination(r_vec, v_vec, mu))

nus_deg = np.array(nus_deg, dtype=float)
incs_deg = np.array(incs_deg, dtype=float)

# ---------- Δv numérico ----------
m_series = X[6, :]
dt = np.diff(tt)
a_inst_series = (T / np.maximum(m_series, 1e-18)) / 1000.0  # km/s^2
delta_v_kms = float(np.sum(a_inst_series[:-1] * dt * (m_series[:-1] > m_dry)))
delta_v_ms  = 1000.0 * delta_v_kms

print("\n=== Dados V-only ===")
print(f"Δv acumulado (m/s):        {delta_v_ms:.6f}")
print(f"Δi sim (último - inicial): {incs_deg[-1] - incs_deg[0]:.9f}")

# ---------- plots ----------
# (1) elementos clássicos (usar 'tt'!)
plot_classic_orbital_elements(tt, orbital_elementss)

# (2) 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
u, vgrid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = earth_radius * np.cos(u) * np.sin(vgrid)
y_e = earth_radius * np.sin(u) * np.sin(vgrid)
z_e = earth_radius * np.cos(vgrid)
ax.plot_wireframe(x_e, y_e, z_e, color="g", alpha=0.3)
ax.plot3D(X[0, :], X[1, :], X[2, :], 'g', label="Satélite V-only")
ax.set_box_aspect([1, 1, 1])
ax.set_title("Órbita simulada - Satélite V-only")
ax.legend()
ax.axis('equal')

# (3) i(ν) segmentado
def plot_i_vs_nu_segmentado(nu_deg: np.ndarray, inc_deg: np.ndarray, *, ax=None,
                            color="black", **plot_kw):
    nu_deg = np.asarray(nu_deg, dtype=float)
    inc_deg = np.asarray(inc_deg, dtype=float)
    if ax is None:
        fig_loc, ax = plt.subplots()
    dn = np.diff(nu_deg)
    wraps = np.where(dn < -180.0)[0]
    start = 0
    for w in wraps:
        ax.plot(nu_deg[start:w+1], inc_deg[start:w+1], color=color, **plot_kw)
        start = w + 1
    ax.plot(nu_deg[start:], inc_deg[start:], color=color, **plot_kw)
    ax.set_xlabel(r'$\nu$ (deg)')
    ax.set_ylabel(r'$i$ (deg)')
    ax.set_xlim(0.0, 360.0)
    ax.grid(True)
    return ax

fig2, ax2 = plt.subplots()
plot_i_vs_nu_segmentado(nus_deg, incs_deg, ax=ax2, color="green", lw=1.5, label='i vs. ν')
ax2.legend()

plt.show()

# ===================== simulate() (mesmo comportamento da seção acima) =====================
def simulate():
    """
    Executa a integração com o x_dot deste módulo e devolve:
      - t: vetor de tempo (s) (a grade pedida pelo usuário)
      - X: estados [x,y,z,vx,vy,vz,m] ao longo do tempo (7 x N)
      - nus_deg: anomalia verdadeira (graus, [0,360))
      - incs_deg: inclinação (graus)
      - orbital_elementss: lista de OrbitalElements por amostra
    """
    x0 = np.concatenate((r, v, [m0]))
    sol = solve_ivp(
        x_dot, (t[0], t[-1]), x0,
        method='RK45', rtol=RTOL, atol=ATOL,
        dense_output=True
    )
    X_ = sol.sol(t)
    orbital_elementss_ = []
    nus_deg_ = []
    incs_deg_ = []
    for k in range(X_.shape[1]):
        r_vec = X_[0:3, k]
        v_vec = X_[3:6, k]
        orbital_elementss_.append(get_orbital_elements(r_vec, v_vec, mu))
        nus_deg_.append(get_true_anormaly(r_vec, v_vec, mu))
        incs_deg_.append(get_inclination(r_vec, v_vec, mu))
    return t, X_, np.array(nus_deg_, dtype=float), np.array(incs_deg_, dtype=float), orbital_elementss_
