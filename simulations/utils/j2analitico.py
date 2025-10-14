import numpy as np
import math
import pandas as pd

# =================== Constantes Terra ===================
MU = 3.986004418e5      # km^3/s^2
RE = 6378.1363          # km
J2 = 1.08262668e-3      # -

# =================== Utilidades ===================
def norm(v): return float(np.linalg.norm(v))

def rv_to_coe(r_km, v_kms, mu=MU):
    r = np.array(r_km, dtype=float)
    v = np.array(v_kms, dtype=float)
    rmag = norm(r); vmag = norm(v)
    h = np.cross(r, v); hmag = norm(h)
    k = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k, h); nmag = norm(n_vec)
    e_vec = (np.cross(v, h) / mu) - (r / rmag); e = norm(e_vec)
    a = -mu / (2.0 * (vmag**2/2.0 - mu/rmag))
    i = math.acos(h[2] / hmag)

    # RAAN Ω
    if nmag > 1e-14:
        Omega = math.acos(n_vec[0] / nmag)
        if n_vec[1] < 0: Omega = 2*math.pi - Omega
    else:
        Omega = 0.0

    # argumento do perigeu ω
    if e > 1e-14 and nmag > 1e-14:
        omega = math.acos(np.dot(n_vec, e_vec) / (nmag * e))
        if e_vec[2] < 0: omega = 2*math.pi - omega
    else:
        omega = 0.0

    # anomalia verdadeira ν (apenas para estado inicial)
    if e > 1e-14:
        nu = math.acos(np.dot(e_vec, r) / (e * rmag))
        if np.dot(r, v) < 0: nu = 2*math.pi - nu
    else:
        if nmag > 1e-14:
            nu = math.acos(np.dot(n_vec, r) / (nmag * rmag))
            if r[2] < 0: nu = 2*math.pi - nu
        else:
            nu = 0.0
    return a, e, i, Omega, omega, nu

def j2_secular_rates(a_km, e, inc_rad):
    """Taxas seculares analíticas (1ª ordem em J2): n, dΩ/dt, dω/dt, du/dt, em rad/s."""
    n  = math.sqrt(MU / a_km**3)
    p  = a_km * (1 - e**2)
    fac = (3.0/2.0) * J2 * (RE**2) / (p**2) * n
    dOmega = -fac * math.cos(inc_rad)
    domega =  0.5 * fac * (5.0*math.cos(inc_rad)**2 - 1.0)
    du     =  n + 0.75 * J2 * (RE**2) / (p**2) * n * (3.0*math.cos(inc_rad)**2 - 1.0)
    return n, dOmega, domega, du

# =================== Condições do Pedro ===================
r0 = np.array([6877.452, 0.0, 0.0])        # km
v0 = np.array([0.0, 5.383, 5.383])         # km/s
t  = np.linspace(0.0, 43200.0, 10000)      # s (12 h)

# Elementos iniciais (medidos do estado)
a, e_measured, inc, Om0, om0, nu0 = rv_to_coe(r0, v0)

# >>>>>>> FORÇAR e = 1e-4 NAS FÓRMULAS ANALÍTICAS <<<<<<<
e_used = 1.0e-4

# Taxas seculares J2 (analítico, 1ª ordem) usando e_used
n, dOm, dom, du = j2_secular_rates(a, e_used, inc)

# Integração analítica dos ângulos (rad)
u0   = om0 + nu0
Om_t = Om0 + dOm * t
om_t = om0 + dom * t
u_t  = u0  + du  * t
nu_t = u_t - om_t     # ν(t) consistente: u - ω

# Conversão para graus (sem saltos 0–360: unwrap)
to_deg = np.degrees
Om_deg = to_deg(np.unwrap(Om_t))
om_deg = to_deg(np.unwrap(om_t))
nu_deg = to_deg(np.unwrap(nu_t))
u_deg  = to_deg(np.unwrap(u_t))

# Impressões de validação
rad_s_to_deg_day = 180.0/math.pi * 86400.0
print("=== Elementos iniciais (medidos de r,v) ===")
print(f"a = {a:.3f} km, e_medido = {e_measured:.6e}, i = {to_deg(inc):.6f} deg")
print(f"e_usado nas taxas = {e_used:.6e}")
print("\n=== Taxas seculares J2 (analíticas, 1ª ordem, com e_usado) ===")
print(f"dΩ/dt = {dOm*rad_s_to_deg_day:.6f} deg/day")
print(f"dω/dt = {dom*rad_s_to_deg_day:.6f} deg/day")
print(f"du/dt = {du *rad_s_to_deg_day:.6f} deg/day   (u = ω + ν)")
print("\n=== Variações esperadas em 12 h ===")
print(f"ΔΩ_12h ≈ {(dOm*rad_s_to_deg_day*0.5):.6f} deg")
print(f"Δω_12h ≈ {(dom*rad_s_to_deg_day*0.5):.6f} deg")

# Salva CSV com a série temporal analítica
df = pd.DataFrame({
    "t_s": t,
    "t_hours": t/3600.0,
    "Omega_deg_unwrapped": Om_deg,
    "omega_deg_unwrapped": om_deg,
    "nu_deg_unwrapped": nu_deg,
    "u_deg_unwrapped": u_deg
})
df.to_csv("J2_analitico_ITASAT2_12h_e1e-4.csv", index=False)
print("\nCSV salvo: J2_analitico_ITASAT2_12h_e1e-4.csv")
