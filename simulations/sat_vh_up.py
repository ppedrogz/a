import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import utils.GetClassicOrbitalElements 


r = np.array([10016.34, -17012.52, 7899.28])
v = np.array([2.5, -1.05, 3.88])
t = np.linspace(0, 2000, 100000)
earth_radius = 6378.0  # in km
mu = 3.986e5
thrust = 2  # N (BIT-3 ~1.1 mN)

# ===================== ADIÇÕES (massa e Busek BIT-3) =====================
# Dados do BIT-3
T   = 1.1e-3       # N (força fixa)
Isp = 2150.0       # s
g0  = 9.80665      # m/s^2

# Estado de massa
m_sat = 20.0       # kg (massa inicial nominal usada no seu aV/aH)
m0    = 20.0       # kg (massa inicial do estado)
m_dry = 15.0       # kg (massa seca)  << já existia e mantida
# ========================================================================

# ===================== ADIÇÕES (H e janelas em anomalia verdadeira) =====================
# OBS: aV/aH fixos foram abandonados; usamos a_inst = (T/m)/1000 dentro de x_dot
THRUST_INTERVAL_DEG = 30.0
MEAN_THETA_LIST_DEG = [180.0]

def throttle(t, x):
    # Sempre ligado enquanto m > m_dry
    return 1.0 if x[6] > m_dry else 0.0

def wrap_deg(a):
    """Normaliza ângulo para [0, 360)."""
    return np.remainder(a, 360.0)

def angle_in_window_deg(theta_deg, center_deg, width_deg):
    """Retorna True se theta estiver dentro da janela centrada em center_deg com largura width_deg."""
    half = 0.5*width_deg
    lo = wrap_deg(center_deg - half)
    hi = wrap_deg(center_deg + half)
    th = wrap_deg(theta_deg)
    if lo <= hi:
        return (th >= lo) and (th <= hi)
    else:
        # janela cruza 0/360
        return (th >= lo) or (th <= hi)

def in_any_window(theta_deg):
    """Combina a lista de janelas e decide o firing."""
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

    # Normas e direção (mantidos)
    vnorm = np.linalg.norm(v_vec)
    h_vec = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h_vec)

    # ---------- PATCH (SUBSTITUIÇÃO): empuxo em RSW ----------
    # motor ligado enquanto houver propelente; módulo da aceleração T/m (km/s^2)
    m_cur = max(x[6], 1e-18)                 # kg (evita div/0)
    u = throttle(t, x)                       # 1.0 se m > m_dry, senão 0.0
    a_inst = (T / m_cur) / 1000.0            # km/s^2 (estado em km e km/s)

    # base RSW: R (radial), S (along-track), W (normal ao plano)
    r_hat = r_vec / (np.linalg.norm(r_vec) + 1e-32)
    h_vec_local = np.cross(r_vec, v_vec)
    w_hat = h_vec_local / (np.linalg.norm(h_vec_local) + 1e-32)   # normal ao plano orbital
    s_hat = np.cross(w_hat, r_hat)                                # tangencial (along-track)
    s_hat /= (np.linalg.norm(s_hat) + 1e-32)

    # janela em ν (anomalia verdadeira)
    theta_deg = get_true_anormaly(r_vec, v_vec, mu)
    fire_H = in_any_window(theta_deg)

    # V sempre ligado; dentro da janela, rotaciona parte para W
    alpha = np.deg2rad(45.0) if fire_H else 0.0   # deflexão desejada (ajustável)
    dir_hat = np.cos(alpha)*s_hat + np.sin(alpha)*w_hat  # |dir_hat|=1

    if u > 0.0:
        xdot[3:6] += a_inst * dir_hat
        # consumo de propelente constante enquanto motor ligado
        xdot[6] = - T/(Isp*g0)
    else:
        xdot[6] = 0.0
    # ---------------------------------------------------------

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
    # anomalia verdadeira (módulo retorna [0,180]); mapeia para (180,360) quando r·v < 0
    nu = get_true_anormaly(r_vec, v_vec, mu)
    if np.dot(r_vec, v_vec) < 0.0:
        nu = (360.0 - nu) % 360.0
    nus_deg.append(nu)
    # inclinação
    incs_deg.append(get_inclination(r_vec, v_vec, mu))

nus_deg = np.array(nus_deg)
incs_deg = np.array(incs_deg)

# ---------- Δv_H e v_apogeu numéricos (várias órbitas) ----------
tt = sol.t
r_norm_series = np.linalg.norm(X[0:3, :].T, axis=1)
v_norm_series = np.linalg.norm(X[3:6, :].T, axis=1)
m_series       = X[6, :]

# janela H (em ν) ao longo de toda a simulação
fire_mask = np.array([in_any_window(nu) for nu in nus_deg], dtype=bool)
# só há thrust se ainda houver propelente
u_mask = (m_series > m_dry)

# integrais por soma de Riemann
dt = np.diff(tt)

# aceleração instantânea (km/s^2) pela força fixa T e massa variável m(t)
a_inst_series = (T / np.maximum(m_series, 1e-18)) / 1000.0  # [km/s^2]

# Δv_H acumulado (apenas quando H está ligado E há propelente)
delta_v_H_kms = float(np.sum(a_inst_series[:-1] * dt * (fire_mask[:-1] & u_mask[:-1])))
delta_v_H_ms  = 1000.0 * delta_v_H_kms  # [m/s]

# "tempo com H ligado" (opcional, só p/ log)
t_H_on = float(np.sum(dt * fire_mask[:-1]))

# v_apogeu numérico: média da velocidade quando H está ativo
if np.any(fire_mask):
    v_apo = float(np.mean(v_norm_series[fire_mask]))
else:
    v_apo = float(np.max(v_norm_series))  # fallback

# Δi_ideal usando v_apo numérico
arg = np.clip(delta_v_H_kms/(2.0*v_apo), -1.0, 1.0)
delta_i_ideal_deg = float(np.degrees(2.0*np.arcsin(arg)))

# Δi simulado (referenciado ao início)
delta_i_sim_deg = incs_deg - incs_deg[0]

print("\n=== Analítico × Simulado (várias órbitas) ===")
print(f"Tempo com H ligado (s):     {t_H_on:.6f}")
print(f"Δv_H acumulado (m/s):       {delta_v_H_ms:.6f}")
print(f"v_apogeu (km/s):            {v_apo:.9f}")
print(f"Δi_ideal (graus):           {delta_i_ideal_deg:.9f}")
print(f"Δi_sim (último - inicial):  {delta_i_sim_deg[-1]:.9f}")

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