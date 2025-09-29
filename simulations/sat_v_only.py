import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from utils.orbitalElementsOperations import *
from utils.visualization import plot_classic_orbital_elements

# ===================== condições iniciais =====================
r= np.array([6890.3, 0, 0]) #parametro da ITASAT 1
v=np.array([0, -0.992,7.535])
#r= np.array([10016.34, -17012.52, 7899.28]) #parametros orbitais iniciais
#v= np.array([2.5, -1.05, 3.88])
t = np.linspace(0, 43200, 100000)  # 5 dias
earth_radius = 6378.0  # km
mu = 3.986e5
thrust = 1.1e-3 # N (força fixa)

# ===================== ADIÇÕES (massa e Busek BIT-3) =====================
T   = thrust    # N (força fixa)
Isp = 2150.0   # s
g0  = 9.80665  # m/s^2

m_sat = 20.0   # kg
m0    = 20.0   # kg
m_dry = 15.0   # kg
# ========================================================================

# adição da perturbação (J2)
from J2 import external_accel, EarthParams, PerturbationFlags

# Flag de módulo para ligar/desligar J2 (não mexe no resto do código)
_J2_ON = False

def _accel_J2(r_vec, v_vec, t):
    if _J2_ON:
        return external_accel(r_vec, v_vec, t,
                              params=EarthParams(),
                              flags=PerturbationFlags(j2=True))
    return 0.0 * r_vec  # vetor zero do mesmo shape

#Perturbação de arrasto atmosferico
from Drag import accel_drag, DragParams
_DRAG_ON = False
_DRAG = DragParams(Cd=2.2, A_ref_m2=0.02, use_atmo_rotation=True,
                   rho0_kg_m3=3.614e-11, h0_km=200.0, H_km=50.0)
def _accel_DRAG(r_vec, v_vec, m_cur):
    return accel_drag(r_vec, v_vec, m_cur, _DRAG) if _DRAG_ON else 0.0 * r_vec

# ===================== Janelas em anomalia verdadeira (não usado aqui) =====================
THRUST_INTERVAL_DEG = 30.0
MEAN_THETA_LIST_DEG = [180] #180 apogeu - 0

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
# ===========================================================================================

def x_dot(t, x):
    xdot = np.zeros_like(x)
    xdot[0:3] = x[3:6]

    # Gravidade
    r_vec = x[0:3]
    v_vec = x[3:6]
    rnorm = np.linalg.norm(r_vec) + 1e-32
    xdot[3:6] = -(mu/(rnorm**3))*r_vec

      # ---------- J2 ----------
    xdot[3:6] += _accel_J2(r_vec, v_vec, t)

    # ----------Arrasto------------
    m_cur = max(x[6], 1e-18)
    xdot[3:6] += _accel_DRAG(r_vec, v_vec, m_cur)

    # ---------- Propulsão só em V ----------
    m_cur = max(x[6], 1e-18)
    u = throttle(t, x)
    a_inst = (T / m_cur) / 1000.0  # km/s^2

    # Base RSW
    r_hat = r_vec / rnorm
    h_vec_local = np.cross(r_vec, v_vec)
    w_hat = h_vec_local / (np.linalg.norm(h_vec_local) + 1e-32)
    s_hat = np.cross(w_hat, r_hat)
    s_hat /= (np.linalg.norm(s_hat) + 1e-32)

    if u > 0.0:
        xdot[3:6] += a_inst * s_hat  # apenas V
        xdot[6] = - T/(Isp*g0)
    else:
        xdot[6] = 0.0
    # ---------------------------------------

    return xdot

# >>> Estado inicial inclui massa <<<
x0 = np.concatenate((r, v, [m0]))

sol = solve_ivp(
    x_dot, (t[0], t[-1]), x0, t_eval=t,
    method="DOP853",       # ou "Radau" (implícito) se preferir
    rtol=1e-12, atol=1e-15,
)
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

    nus_deg.append(nu)

    incs_deg.append(get_inclination(r_vec, v_vec, mu))

nus_deg = np.array(nus_deg, dtype=float)
incs_deg = np.array(incs_deg, dtype=float)

# ---------- Δv numérico ----------
# ---------- Δv acumulado (componentes) ----------
tt = sol.t
m_series = X[6, :]
dt = np.diff(tt)

# Aceleração em V (sempre along-track)
aV_series = (T / np.maximum(m_series, 1e-18)) / 1000.0  # km/s^2

# Mascara "tem propelente"
u_mask = (m_series > m_dry)

# Integrações (somas de Riemann) → km/s
dv_V_kms = float(np.sum(aV_series[:-1] * dt * u_mask[:-1]))
dv_H_kms = 0.0  # não há motor H no V-only

# Converte para m/s
dv_V_ms = 1000.0 * dv_V_kms
dv_H_ms = 0.0
dv_total_ms = dv_V_ms + dv_H_ms

print("\n=== Δv acumulado (V-only) ===")
print(f"Δv_V     (m/s): {dv_V_ms:.6f}")
print(f"Δv_H     (m/s): {dv_H_ms:.6f}")
print(f"Δv_total (m/s): {dv_total_ms:.6f}")

# ---------- utilitário: plot i(nu) segmentado e com cor fixa ----------
def plot_i_vs_nu_segmentado(nu_deg: np.ndarray, inc_deg: np.ndarray, *, ax=None,
                            color="black", **plot_kw):
    """
    Plota i(nu) ligando apenas os pontos dentro de cada trecho entre wraps (360->0),
    forçando uma única cor para todos os segmentos.
    """
    nu_deg = np.asarray(nu_deg, dtype=float)
    inc_deg = np.asarray(inc_deg, dtype=float)

    if ax is None:
        fig, ax = plt.subplots()

    dn = np.diff(nu_deg)
    wraps = np.where(dn < -180.0)[0]  # índice do ponto ANTES do wrap

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

# ---------- plot dos elementos (seu utilitário) ----------
plot_classic_orbital_elements(t, orbital_elementss)

# ---------- plot 3D ----------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Wireframe da Terra
u, vgrid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = earth_radius * np.cos(u) * np.sin(vgrid)
y_e = earth_radius * np.sin(u) * np.sin(vgrid)
z_e = earth_radius * np.cos(vgrid)
ax.plot_wireframe(x_e, y_e, z_e, color="g", alpha=0.3)

# Trajetória simulada (única)
ax.plot3D(X[0, :], X[1, :], X[2, :], 'g', label="Satélite V-only")

ax.set_box_aspect([1, 1, 1])
ax.set_title("Órbita simulada - Satélite V-only")
ax.legend()
ax.axis('equal')

# ---------- plot i(nu) com cor única ----------
fig2, ax2 = plt.subplots()
plot_i_vs_nu_segmentado(nus_deg, incs_deg, ax=ax2, color="green", lw=1.5, label='i vs. nu')
ax2.legend()

plt.show()

def simulate():
    """
    Executa a integração com o x_dot deste módulo e devolve:
    (t, X, nus_deg, incs_deg, orbital_elementss)
    onde:
      - t: vetor de tempo (s)
      - X: estados [x,y,z,vx,vy,vz,m] ao longo do tempo (7 x N)
      - nus_deg: anomalia verdadeira em graus, já em [0, 360)
      - incs_deg: inclinação em graus
      - orbital_elementss: lista de OrbitalElements por amostra
    """
    # estado inicial inclui massa
    x0 = np.concatenate((r, v, [m0]))
    sol = solve_ivp(
    x_dot, (t[0], t[-1]), x0, t_eval=t,
    method="DOP853",       # ou "Radau" (implícito) se preferir
    rtol=1e-12, atol=1e-15,
    )
    X = sol.y

    orbital_elementss = []
    nus_deg = []
    incs_deg = []

    for k in range(X.shape[1]):
        r_vec = X[0:3, k]
        v_vec = X[3:6, k]

        # elementos clássicos completos
        orbital_elementss.append(get_orbital_elements(r_vec, v_vec, mu))

        # anomalia verdadeira JÁ em [0, 360) pela sua get_true_anormaly
        nu = get_true_anormaly(r_vec, v_vec, mu)
        nus_deg.append(nu)

        # inclinação em graus
        incs_deg.append(get_inclination(r_vec, v_vec, mu))

    return sol.t, X, np.array(nus_deg, float), np.array(incs_deg, float), orbital_elementss
def specific_energy(r, v, mu):
    return 0.5*np.dot(v, v) - mu/np.linalg.norm(r)

def a_from_energy(r, v, mu):
    E = specific_energy(r, v, mu)
    return -mu/(2.0*E)

def a_from_pe(r, v, mu):
    h = np.linalg.norm(np.cross(r, v))
    e = np.linalg.norm((np.cross(v, np.cross(r, v))/mu) - r/np.linalg.norm(r))
    p = h*h/mu
    return p/(1.0 - e*e)

a_E  = np.array([a_from_energy(X[0:3,k], X[3:6,k], mu) for k in range(X.shape[1])])
a_pe = np.array([a_from_pe    (X[0:3,k], X[3:6,k], mu) for k in range(X.shape[1])])

print("drift(a_E)  [km] =", a_E.max()-a_E.min())
print("drift(a_pe) [km] =", a_pe.max()-a_pe.min())
print("max |a_E - a_pe| [m] =", 1000*np.max(np.abs(a_E-a_pe)))