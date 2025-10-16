import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# =================== Constantes Terra ===================
MU = 3.986004418e5      # km^3/s^2
RE = 6378.1363          # km
J2 = 1.08262668e-3      # -
OMEGA_E = 7.2921159e-5  # rad/s (rotação da Terra)

# Tesseral grau-2 ordem-2
C22 = 1.574e-6
S22 = -0.903e-6
J22_SCALAR = 1.766e-6   # módulo alternativo (opcional)

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

    # anomalia verdadeira ν
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
    n  = math.sqrt(MU / a_km**3)
    p  = a_km * (1 - e**2)
    fac = (3.0/2.0) * J2 * (RE**2) / (p**2) * n
    dOmega = -fac * math.cos(inc_rad)
    domega =  0.5 * fac * (5.0*math.cos(inc_rad)**2 - 1.0)
    du     =  n + 0.75 * J2 * (RE**2) / (p**2) * n * (3.0*math.cos(inc_rad)**2 - 1.0)
    return n, dOmega, domega, du

# =================== Bloco J22 (tesseral 2,2) ===================
def j22_long_period_rates(t_array, a_km, e, inc_rad, Om_t, om_t, M0,
                          theta_g0=0.0,
                          use_C22_S22=True,
                          C22_val=C22, S22_val=S22,
                          J22_val=J22_SCALAR):
    """
    Incrementos de longo-período (rad) de Ω, ω, u e i devido a J22 ao longo do tempo.
    >>> Substitua as amplitudes A_* e as fases para as expressões do seu PPT, se quiser exatidão.
    """
    t = np.asarray(t_array, dtype=float)
    n = math.sqrt(MU / a_km**3)
    p = a_km * (1 - e**2)

    # Fase tesseral (aprox. não-resonante)
    psi_t = theta_g0 + OMEGA_E*t - Om_t - om_t - (M0 + n*t)

    # Coeficiente tesseral (C22/S22 ou módulo J22)
    if use_C22_S22:
        K22 = C22_val*np.cos(2*psi_t) + S22_val*np.sin(2*psi_t)
    else:
        K22 = J22_val * np.cos(2*psi_t)  # ajuste fase se necessário

    # Amplitudes genéricas (placeholders). Cole as fórmulas exatas do PPT aqui:
    fac = (RE/p)**2
    ci = np.cos(inc_rad); si = np.sin(inc_rad)

    # Dependências típicas para l=2,m=2 (apenas para dar ordem de grandeza estável):
    A_Om =  0.75 * fac * si*si
    A_om =  0.75 * fac * (1 + ci*ci)
    A_u  =  0.75 * fac * (3*ci*ci - 1)
    A_i  =  1.50 * fac * si*ci  # <- inclinação (placeholder); substitua pelo do slide

    # Denominador de não-ressonância (ordem de grandeza)
    delta = (n - OMEGA_E)
    delta = delta if abs(delta) > 1e-10 else 1e-10

    # Integrais de longo-período (formas seno/cosseno são placeholders — ajuste conforme o PPT)
    dOm22 =  (A_Om * K22) * np.sin(2*psi_t) / (2.0*delta)
    dom22 =  (A_om * K22) * np.cos(2*psi_t) / (2.0*delta)
    du22  =  (A_u  * K22) * np.sin(2*psi_t) / (2.0*delta)
    di22  =  (A_i  * K22) * np.cos(2*psi_t) / (2.0*delta)

    return dOm22, dom22, du22, di22

# =================== Condições iniciais ===================
r0 = np.array([6877.452, 0.0, 0.0])        # km
v0 = np.array([0.0, 5.383, 5.383])         # km/s
t  = np.linspace(0.0, 43200.0, 10000)      # s (12 h)

a, e_measured, inc, Om0, om0, nu0 = rv_to_coe(r0, v0)
e_used = 1.0e-4

# J2: séries base
n, dOm, dom, du = j2_secular_rates(a, e_used, inc)
u0   = om0 + nu0
Om_t = Om0 + dOm * t
om_t = om0 + dom * t
u_t  = u0  + du  * t
nu_t = u_t - om_t
i_t  = np.full_like(t, inc)  # J2 1ª ordem não altera i

# J22: incrementos
M0 = nu0  # para e pequeno
dOm22, dom22, du22, di22 = j22_long_period_rates(t, a, e_used, inc, Om_t, om_t, M0)

# Séries completas J2+J22
Om_t_J2J22 = Om_t + dOm22
om_t_J2J22 = om_t + dom22
u_t_J2J22  = u_t  + du22
nu_t_J2J22 = u_t_J2J22 - om_t_J2J22
i_t_J2J22  = i_t  + di22

# Conversão para graus (unwrap para ângulos cíclicos)
Om_deg, om_deg, nu_deg, u_deg = map(np.degrees, map(np.unwrap, [Om_t, om_t, nu_t, u_t]))
Om_deg_J2J22, om_deg_J2J22, nu_deg_J2J22, u_deg_J2J22 = map(np.degrees, map(np.unwrap, [Om_t_J2J22, om_t_J2J22, nu_t_J2J22, u_t_J2J22]))
# Inclinação (não precisa unwrap)
i_deg       = np.degrees(i_t)
i_deg_J2J22 = np.degrees(i_t_J2J22)

# Impressões J2
rad_s_to_deg_day = 180.0/math.pi * 86400.0
print("=== Elementos iniciais (medidos de r,v) ===")
print(f"a = {a:.3f} km, e_medido = {e_measured:.6e}, i = {to_deg(inc):.6f} deg")
print(f"e_usado nas taxas = {e_used:.6e}")
print("\n=== Taxas seculares J2 (analíticas, 1ª ordem) ===")
print(f"dΩ/dt = {dOm*rad_s_to_deg_day:.6f} deg/day")
print(f"dω/dt = {dom*rad_s_to_deg_day:.6f} deg/day")
print(f"du/dt = {du *rad_s_to_deg_day:.6f} deg/day")
print("\n=== Variações esperadas em 12 h ===")
print(f"ΔΩ_12h ≈ {(dOm*rad_s_to_deg_day*0.5):.6f} deg")
print(f"Δω_12h ≈ {(dom*rad_s_to_deg_day*0.5):.6f} deg")

# ======================= RELATÓRIO (efeito J22 isolado) =======================
# deltas em arcseg para Ω, ω, u, ν
deg_to_arcsec = 3600.0
dOm_arcsec = (Om_deg_J2J22 - Om_deg) * deg_to_arcsec
dom_arcsec = (om_deg_J2J22 - om_deg) * deg_to_arcsec
du_arcsec  = (u_deg_J2J22  - u_deg)  * deg_to_arcsec
dnu_arcsec = (nu_deg_J2J22 - nu_deg) * deg_to_arcsec
# delta de inclinação em arcseg
di_arcsec  = (i_deg_J2J22 - i_deg) * deg_to_arcsec

# variações máximas (máx - mín) em graus e arcseg
dOm22_max_deg = (np.max(Om_deg_J2J22 - Om_deg) - np.min(Om_deg_J2J22 - Om_deg))
dom22_max_deg = (np.max(om_deg_J2J22 - om_deg) - np.min(om_deg_J2J22 - om_deg))
du22_max_deg  = (np.max(u_deg_J2J22  - u_deg)  - np.min(u_deg_J2J22  - u_deg))
dnu22_max_deg = (np.max(nu_deg_J2J22 - nu_deg) - np.min(nu_deg_J2J22 - nu_deg))
di22_max_deg  = (np.max(i_deg_J2J22  - i_deg)  - np.min(i_deg_J2J22  - i_deg))

print("\n=== Variações de longo-período devido a J22 (em 12 h) ===")
print(f"Δi_J22  ≈ {di22_max_deg:.6e} deg  ({di22_max_deg*deg_to_arcsec:.3f} arcseg)")
print(f"ΔΩ_J22  ≈ {dOm22_max_deg:.6e} deg  ({dOm22_max_deg*deg_to_arcsec:.3f} arcseg)")
print(f"Δω_J22  ≈ {dom22_max_deg:.6e} deg  ({dom22_max_deg*deg_to_arcsec:.3f} arcseg)")
print(f"Δν_J22  ≈ {dnu22_max_deg:.6e} deg  ({dnu22_max_deg*deg_to_arcsec:.3f} arcseg)")

# ======================= PLOTS (J2 vs J2+J22) =======================
# Fig 1: séries absolutas (em graus) — Ω, ω, u, ν
fig1, axs1 = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
axs1 = axs1.ravel()
axs1[0].plot(t/3600.0, Om_deg, label='Ω (J2)', lw=1.6)
axs1[0].plot(t/3600.0, Om_deg_J2J22, '--', label='Ω (J2+J22)', lw=1.2)
axs1[0].set_title('RAAN Ω (graus)'); axs1[0].set_xlabel('Tempo (h)'); axs1[0].grid(True); axs1[0].legend()

axs1[1].plot(t/3600.0, om_deg, label='ω (J2)', lw=1.6)
axs1[1].plot(t/3600.0, om_deg_J2J22, '--', label='ω (J2+J22)', lw=1.2)
axs1[1].set_title('Arg. do perigeu ω (graus)'); axs1[1].set_xlabel('Tempo (h)'); axs1[1].grid(True); axs1[1].legend()

axs1[2].plot(t/3600.0, u_deg, label='u (J2)', lw=1.6)
axs1[2].plot(t/3600.0, u_deg_J2J22, '--', label='u (J2+J22)', lw=1.2)
axs1[2].set_title('Argumento de latitude u (graus)'); axs1[2].set_xlabel('Tempo (h)'); axs1[2].grid(True); axs1[2].legend()

axs1[3].plot(t/3600.0, nu_deg, label='ν (J2)', lw=1.6)
axs1[3].plot(t/3600.0, nu_deg_J2J22, '--', label='ν (J2+J22)', lw=1.2)
axs1[3].set_title('Anomalia verdadeira ν (graus)'); axs1[3].set_xlabel('Tempo (h)'); axs1[3].grid(True); axs1[3].legend()

plt.suptitle('Séries absolutas: J2 vs J2+J22', y=1.02)

# Fig 2: deltas (efeito J22 isolado) em arcseg — Ω, ω, u, ν
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
axs2 = axs2.ravel()
axs2[0].plot(t/3600.0, dOm_arcsec, lw=1.8)
axs2[0].set_title('ΔΩ devido a J22 (arcseg)'); axs2[0].set_xlabel('Tempo (h)'); axs2[0].grid(True)

axs2[1].plot(t/3600.0, dom_arcsec, lw=1.8)
axs2[1].set_title('Δω devido a J22 (arcseg)'); axs2[1].set_xlabel('Tempo (h)'); axs2[1].grid(True)

axs2[2].plot(t/3600.0, du_arcsec, lw=1.8)
axs2[2].set_title('Δu devido a J22 (arcseg)'); axs2[2].set_xlabel('Tempo (h)'); axs2[2].grid(True)

axs2[3].plot(t/3600.0, dnu_arcsec, lw=1.8)
axs2[3].set_title('Δν devido a J22 (arcseg)'); axs2[3].set_xlabel('Tempo (h)'); axs2[3].grid(True)

plt.suptitle('Efeito isolado de J22 (arcseg)', y=1.02)

# Fig extra: delta de inclinação em arcseg
fig3, ax3 = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
ax3.plot(t/3600.0, di_arcsec, lw=1.8)
ax3.set_title('Δi devido a J22 (arcseg)'); ax3.set_xlabel('Tempo (h)'); ax3.grid(True)

# Relatório rápido no terminal (máximos absolutos dos deltas instantâneos)
print("\n=== Magnitudes (máximos absolutos) do efeito J22 em 12 h (arcseg, instantâneo) ===")
print(f"max |Δi|  = {np.max(np.abs(di_arcsec)):.3e} arcseg")
print(f"max |ΔΩ|  = {np.max(np.abs(dOm_arcsec)):.3e} arcseg")
print(f"max |Δω|  = {np.max(np.abs(dom_arcsec)):.3e} arcseg")
print(f"max |Δu|  = {np.max(np.abs(du_arcsec)):.3e} arcseg")
print(f"max |Δν|  = {np.max(np.abs(dnu_arcsec)):.3e} arcseg")

plt.show()

# Salva CSV
df = pd.DataFrame({
    "t_s": t,
    "t_hours": t/3600.0,
    "i_deg": i_deg,
    "Omega_deg_unwrapped": Om_deg,
    "omega_deg_unwrapped": om_deg,
    "nu_deg_unwrapped": nu_deg,
    "u_deg_unwrapped": u_deg,
    "i_deg_J2J22": i_deg_J2J22,
    "Omega_deg_unwrapped_J2J22": Om_deg_J2J22,
    "omega_deg_unwrapped_J2J22": om_deg_J2J22,
    "nu_deg_unwrapped_J2J22": nu_deg_J2J22,
    "u_deg_unwrapped_J2J22": u_deg_J2J22
})
df.to_csv("J2J22_analitico_ITASAT2_12h.csv", index=False)
print("\nCSV salvo: J2J22_analitico_ITASAT2_12h.csv")
