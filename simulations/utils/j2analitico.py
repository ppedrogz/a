import numpy as np
import math
import pandas as pd

# =================== Constantes Terra ===================
MU = 3.986004418e5      # km^3/s^2
RE = 6378.1363          # km
J2 = 1.08262668e-3      # -
OMEGA_E = 7.2921159e-5  # rad/s (rotação da Terra)

# Tesseral grau-2 ordem-2 (EGM típico; pode ajustar)
C22 = 1.574e-6
S22 = -0.903e-6

# =================== Utilidades ===================
def norm(v): return float(np.linalg.norm(v))
to_deg = np.degrees

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

# =================== Incrementos analíticos J22 (tesseral 2,2) ===================
def j22_long_period_increments(t_array, a_km, e, inc_rad, Om_t, om_t, M0,
                               theta_g0=0.0,
                               C22_val=C22, S22_val=S22):
    """
    Incrementos analíticos (rad) de Ω, ω, u e i devido a J22 ao longo do tempo (não-ressonante).
    Fórmulas (Kaula l=2,m=2) integradas assumindo ψ' ≈ Δ = Ω_E - n constante (1ª ordem).

    Retorna: dOm22(t), dom22(t), du22(t), di22(t)
    """
    t = np.asarray(t_array, dtype=float)
    n = math.sqrt(MU / a_km**3)
    p = a_km * (1 - e**2)

    # Fase tesseral
    psi_t = theta_g0 + OMEGA_E*t - Om_t - om_t - (M0 + n*t)
    Delta = (OMEGA_E - n)
    if abs(Delta) < 1e-12:  # evita divisão por zero (quase-ressonância)
        Delta = np.sign(Delta) * 1e-12 if Delta != 0.0 else 1e-12

    # Coeficiente dimensional
    K = 3.0 * n * (RE/p)**2

    # Amplitudes (Kaula l=2, m=2)
    ci = np.cos(inc_rad); si = np.sin(inc_rad)
    A_Om = -0.5 * K * (si**2)                 # dΩ/dt
    A_om =  0.25 * K * (5.0*ci**2 - 1.0)      # dω/dt
    A_u  =  0.25 * K * (3.0*ci**2 - 1.0)      # du/dt
    B_i  = -0.5 * K * (si*ci)                 # di/dt com (C sin2ψ - S cos2ψ)

    # Integrais:
    # ∫[C cos2ψ + S sin2ψ] dt = [C sin2ψ - S cos2ψ]/(2Δ)
    common_CS = (C22_val*np.sin(2.0*psi_t) - S22_val*np.cos(2.0*psi_t)) / (2.0*Delta)

    # ∫[C sin2ψ - S cos2ψ] dt = [ -C cos2ψ - S sin2ψ ]/(2Δ)
    common_SmC = ( -C22_val*np.cos(2.0*psi_t) - S22_val*np.sin(2.0*psi_t) ) / (2.0*Delta)

    dOm22 = A_Om * common_CS
    dom22 = A_om * common_CS
    du22  = A_u  * common_CS
    di22  = B_i  * common_SmC

    return dOm22, dom22, du22, di22

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

# Integração analítica dos ângulos (rad) — componente J2
u0   = om0 + nu0
Om_t = Om0 + dOm * t
om_t = om0 + dom * t
u_t  = u0  + du  * t
nu_t = u_t - om_t     # ν(t) consistente: u - ω
i_t  = np.full_like(t, inc)  # J2 (1ª ordem) não altera i

# ============== J22 (tesseral 2,2): incrementos long-period analíticos (Kaula)
# Anomalia média inicial (aprox para e pequeno): M0 ≈ ν0 (pode refinar se quiser)
M0 = nu0
dOm22, dom22, du22, di22 = j22_long_period_increments(
    t_array=t, a_km=a, e=e_used, inc_rad=inc,
    Om_t=Om_t, om_t=om_t, M0=M0,
    theta_g0=0.0, C22_val=C22, S22_val=S22
)

# Séries completas J2+J22
Om_t_J2J22 = Om_t + dOm22
om_t_J2J22 = om_t + dom22
u_t_J2J22  = u_t  + du22
nu_t_J2J22 = u_t_J2J22 - om_t_J2J22
i_t_J2J22  = i_t  + di22

# Conversão para graus (unwrap onde é angular cíclico)
Om_deg      = np.degrees(np.unwrap(Om_t))
om_deg      = np.degrees(np.unwrap(om_t))
u_deg       = np.degrees(np.unwrap(u_t))
nu_deg      = np.degrees(np.unwrap(nu_t))
i_deg       = np.degrees(i_t)              # não cíclico

Om_deg_J2J22 = np.degrees(np.unwrap(Om_t_J2J22))
om_deg_J2J22 = np.degrees(np.unwrap(om_t_J2J22))
u_deg_J2J22  = np.degrees(np.unwrap(u_t_J2J22))
nu_deg_J2J22 = np.degrees(np.unwrap(nu_t_J2J22))
i_deg_J2J22  = np.degrees(i_t_J2J22)

# Impressões de validação (J2)
rad_s_to_deg_day = 180.0/math.pi * 86400.0
print("=== Elementos iniciais (medidos de r,v) ===")
print(f"a = {a:.3f} km, e_medido = {e_measured:.6e}, i = {to_deg(inc):.6f} deg")
print(f"e_usado nas taxas = {e_used:.6e}")
print("\n=== Taxas seculares J2 (analíticas, 1ª ordem) ===")
print(f"dΩ/dt = {dOm*rad_s_to_deg_day:.6f} deg/day")
print(f"dω/dt = {dom*rad_s_to_deg_day:.6f} deg/day")
print(f"du/dt = {du *rad_s_to_deg_day:.6f} deg/day")
print("\n=== Variações esperadas em 12 h (apenas J2) ===")
print(f"ΔΩ_12h ≈ {(dOm*rad_s_to_deg_day*0.5):.6f} deg")
print(f"Δω_12h ≈ {(dom*rad_s_to_deg_day*0.5):.6f} deg")

# ======================= RELATÓRIO (efeito J22 isolado) =======================
# deltas instantâneos (J2+J22 - J2), em graus
dOm_deg = Om_deg_J2J22 - Om_deg
dom_deg = om_deg_J2J22 - om_deg
du_deg  = u_deg_J2J22  - u_deg
dnu_deg = nu_deg_J2J22 - nu_deg
di_deg  = i_deg_J2J22  - i_deg

# variações máximas (máx - mín) em graus e arcseg
deg_to_arcsec = 3600.0
dOm22_max_deg = np.max(dOm_deg) - np.min(dOm_deg)
dom22_max_deg = np.max(dom_deg) - np.min(dom_deg)
du22_max_deg  = np.max(du_deg)  - np.min(du_deg)
dnu22_max_deg = np.max(dnu_deg) - np.min(dnu_deg)
di22_max_deg  = np.max(di_deg)  - np.min(di_deg)

print("\n=== Variações de longo-período devidas a J22 (em 12 h) ===")
print(f"Δi_J22  ≈ {di22_max_deg:.6e} deg  ({di22_max_deg*deg_to_arcsec:.3f} arcseg)")
print(f"ΔΩ_J22  ≈ {dOm22_max_deg:.6e} deg  ({dOm22_max_deg*deg_to_arcsec:.3f} arcseg)")
print(f"Δω_J22  ≈ {dom22_max_deg:.6e} deg  ({dom22_max_deg*deg_to_arcsec:.3f} arcseg)")
print(f"Δν_J22  ≈ {dnu22_max_deg:.6e} deg  ({dnu22_max_deg*deg_to_arcsec:.3f} arcseg)")

# ======================= CSV (comparação J2 vs J2+J22) =======================
df = pd.DataFrame({
    "t_s": t,
    "t_hours": t/3600.0,
    "i_deg_J2": i_deg,
    "Omega_deg_J2": Om_deg,
    "omega_deg_J2": om_deg,
    "u_deg_J2": u_deg,
    "nu_deg_J2": nu_deg,
    "i_deg_J2J22": i_deg_J2J22,
    "Omega_deg_J2J22": Om_deg_J2J22,
    "omega_deg_J2J22": om_deg_J2J22,
    "u_deg_J2J22": u_deg_J2J22,
    "nu_deg_J2J22": nu_deg_J2J22,
    "delta_i_deg": di_deg,
    "delta_Omega_deg": dOm_deg,
    "delta_omega_deg": dom_deg,
    "delta_u_deg": du_deg,
    "delta_nu_deg": dnu_deg
})
df.to_csv("J2J22_analitico_ITASAT2_12h.csv", index=False)
print("\nCSV salvo: J2J22_analitico_ITASAT2_12h.csv")
