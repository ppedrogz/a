# main_constelacao.py
# Execução consolidada das três simulações (VH UP, V ONLY, VH DOWN)
# + diagnósticos de elementos
# + gráficos individuais COM vs SEM J2
# + gráficos combinados COM vs SEM J2
# Observação: requer que os módulos sat_vh_up/sat_v_only/sat_vh_down aceitem simulate(j2: bool)

import numpy as np
import matplotlib.pyplot as plt

# seus módulos de simulação
import sat_vh_up as sat_vh_up
import sat_v_only as sat_v_only
import sat_vh_down as sat_vh_down

from J2 import EarthParams, diagnose_orbital_changes
from utils.GetClassicOrbitalElements import (
    get_orbital_elements,
    get_true_anormaly,   # mantém o nome da sua função
    get_inclination
)

# ----------------- Helpers visuais -----------------
def _unwrap_deg(a_deg):
    """Desenrola série angular em graus (evita saltos 360->0)."""
    return np.degrees(np.unwrap(np.radians(a_deg)))

def _plot_i_vs_nu_segmentado(nu_deg, inc_deg, *, ax=None, color="k", label=None, **kw):
    """Plota i(ν) em segmentos entre wraps para não ligar 360->0, mantendo cor única."""
    nu_deg = np.asarray(nu_deg, float)
    inc_deg = np.asarray(inc_deg, float)
    if ax is None:
        fig, ax = plt.subplots()
    dn = np.diff(nu_deg)
    wraps = np.where(dn < -180.0)[0]
    start = 0
    first = True
    for w in wraps:
        ax.plot(nu_deg[start:w+1], inc_deg[start:w+1], color=color,
                label=(label if first else None), **kw)
        first = False
        start = w + 1
    ax.plot(nu_deg[start:], inc_deg[start:], color=color,
            label=(label if first else None), **kw)
    ax.set_xlim(0, 360)
    ax.set_xlabel("Anomalia Verdadeira ν [graus]")
    ax.set_ylabel("Inclinação i [graus]")
    ax.grid(alpha=0.3)
    return ax

def _elements_series(t, X, mu):
    """Séries de elementos clássicos: ν, i, Ω, ω, e, a (em graus onde apropriado)."""
    N = X.shape[1]
    nus = np.empty(N); incs = np.empty(N)
    RAAN = np.empty(N); argp = np.empty(N)
    ecc = np.empty(N);  sma  = np.empty(N)
    for k in range(N):
        r = X[0:3, k]; v = X[3:6, k]
        el = get_orbital_elements(r, v, mu)
        nus[k]  = get_true_anormaly(r, v, mu)   # sua função já em [0,360)
        incs[k] = el.inclination
        RAAN[k] = el.ascending_node
        argp[k] = el.argument_of_perigee
        ecc[k]  = el.eccentricity
        sma[k]  = el.major_axis
    return nus, incs, RAAN, argp, ecc, sma

def _call_sim_with_flag(module, flag: bool):
    """
    Chama module.simulate(j2=flag). Se não existir, tenta outras assinaturas.
    Se nada servir, retorna None (caller pode usar fallback).
    """
    try:
        return module.simulate(j2=flag)
    except TypeError:
        pass
    try:
        return module.simulate(use_perturbations=flag)
    except TypeError:
        pass
    try:
        return module.simulate(j2=flag)  # redundância defensiva
    except TypeError:
        pass
    return None

def _two_body_propagate(x0, t, mu):
    """
    Fallback SEM PERTURBAÇÃO: 2-corpos (sem J2/empuxo). Mantém massa constante se existir.
    (Só usado se o módulo NÃO aceitar j2=, o que NÃO é o ideal.)
    """
    from scipy.integrate import solve_ivp
    n = len(x0)
    _has_mass = n >= 7
    def f(_t, x):
        r = x[0:3]; v = x[3:6]
        rn = np.linalg.norm(r) + 1e-32
        a = -(mu/rn**3)*r
        dx = np.zeros_like(x)
        dx[0:3] = v
        dx[3:6] = a
        if _has_mass:
            dx[6] = 0.0
        return dx
    sol = solve_ivp(f, (t[0], t[-1]), x0, t_eval=t, method='RK45')
    return sol.y

def _sync_series(t_ref, t_src, arr_src):
    """
    Interpola arr_src para o tempo de referência t_ref.
    arr_src pode ser (N,) ou (M,N), onde N = len(t_src).
    """
    from numpy import interp
    t_ref = np.asarray(t_ref)
    t_src = np.asarray(t_src)
    arr_src = np.asarray(arr_src)
    if arr_src.ndim == 1:
        return interp(t_ref, t_src, arr_src)
    # estados (linhas) ou várias séries em paralelo
    out = np.empty((arr_src.shape[0], t_ref.size), dtype=arr_src.dtype)
    for i in range(arr_src.shape[0]):
        out[i, :] = interp(t_ref, t_src, arr_src[i, :])
    return out

def _plot_satellite_3d_comparison(name, X_on, X_off, RE, color="#1f77b4"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Terra
    u, vgrid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x_e = RE * np.cos(u)*np.sin(vgrid)
    y_e = RE * np.sin(u)*np.sin(vgrid)
    z_e = RE * np.cos(vgrid)
    ax.plot_wireframe(x_e, y_e, z_e, color="g", alpha=0.25)
    ax.set_box_aspect([1,1,1])

    # Trajetórias
    ax.plot3D(X_on[0,:],  X_on[1,:],  X_on[2,:],  '-',  color=color, lw=1.6, label=f"{name} (J2 ON)")
    ax.plot3D(X_off[0,:], X_off[1,:], X_off[2,:], '--', color=color, lw=1.6, label=f"{name} (J2 OFF)")

    ax.set_title(f"{name}: Trajetória 3D — COM (linha cheia) × SEM J2 (tracejada)")
    ax.legend()
    ax.axis('equal')
    plt.show()

def _plot_satellite_i_vs_nu_comparison(name, Nu_on, I_on, Nu_off, I_off, color="#1f77b4"):
    fig, ax = plt.subplots()
    _plot_i_vs_nu_segmentado(Nu_on,  I_on,  ax=ax, color=color,      lw=1.6, label=f"{name} (J2 ON)")
    _plot_i_vs_nu_segmentado(Nu_off, I_off, ax=ax, color=color,      lw=1.6, label=f"{name} (J2 OFF)", alpha=0.35)
    ax.set_title(f"{name}: Inclinação i(ν) — COM × SEM J2")
    ax.legend()
    plt.show()

# ----------------- Simulações ON (como você já fazia) -----------------
print("Simulando satélite VH UP...")
res_up = sat_vh_up.simulate()  # padrão j2=True dentro do módulo

print("Simulando satélite V ONLY...")
res_v = sat_v_only.simulate()

print("Simulando satélite VH DOWN...")
res_down = sat_vh_down.simulate()

t_up, X_up, nus_up, incs_up, elems_up = res_up
t_v, X_v, nus_v, incs_v, elems_v = res_v
t_down, X_down, nus_down, incs_down, elems_down = res_down

# (opcional) reexecuta para garantir dados frescos para os plots comparativos
t_up, X_up, nus_up, incs_up, elems_up = sat_vh_up.simulate()
t_v, X_v, nus_v, incs_v, elems_v = sat_v_only.simulate()
t_down, X_down, nus_down, incs_down, elems_down = sat_vh_down.simulate()

# ----------------- Diagnósticos (J2 ON) -----------------
P = EarthParams()  # (mu, Re, J2)
earth_radius = getattr(P, "Re", 6378.0)
MU = getattr(P, "mu", 3.986e5)

print("\n##### Diagnósticos de elementos (efeitos compatíveis com J2) #####")
diag_up   = diagnose_orbital_changes(t_up,   X_up,   params=P, label="VH UP")
diag_v    = diagnose_orbital_changes(t_v,    X_v,    params=P, label="V ONLY")
diag_down = diagnose_orbital_changes(t_down, X_down, params=P, label="VH DOWN")

# ----------------- Gráfico 3D (COM J2) -----------------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

u, vgrid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = earth_radius * np.cos(u)*np.sin(vgrid)
y_e = earth_radius * np.sin(u)*np.sin(vgrid)
z_e = earth_radius * np.cos(vgrid)
ax.plot_wireframe(x_e, y_e, z_e, color="g", alpha=0.3)
ax.set_box_aspect([1, 1, 1])

ax.plot3D(X_up[0, :],   X_up[1, :],   X_up[2, :],   'b-', label="VH Up (J2 ON)")
ax.plot3D(X_v[0, :],    X_v[1, :],    X_v[2, :],    'g-', label="V Only (J2 ON)")
ax.plot3D(X_down[0, :], X_down[1, :], X_down[2, :], 'r-', label="VH Down (J2 ON)")

ax.set_title("Constelação - 3 Satélites (J2 ON)")
ax.legend()
ax.axis('equal')
plt.show()

# ----------------- i(ν) (COM J2) -----------------
plt.figure()
plt.plot(nus_up,   incs_up,   'b.', ms=0.7, label="VH Up")
plt.plot(nus_v,    incs_v,    'g.', ms=0.7, label="V Only")
plt.plot(nus_down, incs_down, 'r.', ms=0.7, label="VH Down")
plt.xlim(0, 360)
plt.xlabel("Anomalia Verdadeira ν [graus]")
plt.ylabel("Inclinação i [graus]")
plt.title("Inclinação vs ν - Comparação dos Satélites (J2 ON)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ----------------- Massa vs tempo (COM J2) -----------------
plt.figure()
plt.plot(t_up,   X_up[6, :],   'b-', label="VH Up")
plt.plot(t_v,    X_v[6, :],    'g-', label="V Only")
plt.plot(t_down, X_down[6, :], 'r-', label="VH Down")
plt.xlabel("Tempo [s]")
plt.ylabel("Massa [kg]")
plt.title("Consumo de Propelente (J2 ON)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# =====================================================================
# ADENDO: COMPARAÇÃO COM x SEM J2 (gráficos de influência)
# =====================================================================
print("\n##### Adendo: Comparação COM x SEM J2 (gráficos de influência) #####")

# --------- séries COM perturbação (recalcula Ω, ω para gráficos J2) ---------
print("Computando séries COM perturbação para gráficos J2...")
Nu_up_on,  I_up_on,  Om_up_on,  W_up_on,  e_up_on,  a_up_on  = _elements_series(t_up,   X_up,   MU)
Nu_v_on,   I_v_on,   Om_v_on,   W_v_on,   e_v_on,   a_v_on   = _elements_series(t_v,    X_v,    MU)
Nu_dn_on,  I_dn_on,  Om_dn_on,  W_dn_on,  e_dn_on,  a_dn_on  = _elements_series(t_down, X_down, MU)

# --------- construir referências SEM perturbação (J2 OFF) ---------
print("Gerando referências SEM perturbação (J2 OFF) usando os próprios módulos...")
_res_up_off  = _call_sim_with_flag(sat_vh_up,   False)
_res_v_off   = _call_sim_with_flag(sat_v_only,  False)
_res_dn_off  = _call_sim_with_flag(sat_vh_down, False)

if _res_up_off is None:
    # Fallback (não recomendado): 2-corpos sem thrust
    X_up_off  = _two_body_propagate(X_up[:, 0].copy(), t_up, MU)
    Nu_up_off, I_up_off, Om_up_off, W_up_off, e_up_off, a_up_off = _elements_series(t_up,   X_up_off,  MU)
else:
    t_up_off, X_up_off, Nu_up_off, I_up_off, _ = _res_up_off
    # se tempos diferirem, interpola OFF em t_up
    if not np.allclose(t_up, t_up_off):
        X_up_off  = _sync_series(t_up, t_up_off, X_up_off)
        Nu_up_off = _sync_series(t_up, t_up_off, Nu_up_off)
        I_up_off  = _sync_series(t_up, t_up_off, I_up_off)
    # recalcula Ω, ω, e, a nas amostras alinhadas
    _, _, Om_up_off, W_up_off, e_up_off, a_up_off = _elements_series(t_up, X_up_off, MU)

if _res_v_off is None:
    X_v_off   = _two_body_propagate(X_v[:, 0].copy(),  t_v, MU)
    Nu_v_off,  I_v_off,  Om_v_off,  W_v_off,  e_v_off,  a_v_off  = _elements_series(t_v,    X_v_off,   MU)
else:
    t_v_off, X_v_off, Nu_v_off, I_v_off, _ = _res_v_off
    if not np.allclose(t_v, t_v_off):
        X_v_off  = _sync_series(t_v, t_v_off, X_v_off)
        Nu_v_off = _sync_series(t_v, t_v_off, Nu_v_off)
        I_v_off  = _sync_series(t_v, t_v_off, I_v_off)
    _, _, Om_v_off, W_v_off, e_v_off, a_v_off = _elements_series(t_v, X_v_off, MU)

if _res_dn_off is None:
    X_dn_off  = _two_body_propagate(X_down[:, 0].copy(), t_down, MU)
    Nu_dn_off, I_dn_off, Om_dn_off, W_dn_off, e_dn_off, a_dn_off = _elements_series(t_down, X_dn_off,  MU)
else:
    t_dn_off, X_dn_off, Nu_dn_off, I_dn_off, _ = _res_dn_off
    if not np.allclose(t_down, t_dn_off):
        X_dn_off  = _sync_series(t_down, t_dn_off, X_dn_off)
        Nu_dn_off = _sync_series(t_down, t_dn_off, Nu_dn_off)
        I_dn_off  = _sync_series(t_down, t_dn_off, I_dn_off)
    _, _, Om_dn_off, W_dn_off, e_dn_off, a_dn_off = _elements_series(t_down, X_dn_off, MU)

# =================== TABELA: Ω̇(J2 teórico) × Ω̇(numérico) ===================

def j2_raan_rate_theory(a_km, e, i_deg, mu_km3s2, Re_km, J2):
    # dΩ/dt = -(3/2) J2 (Re/p)^2 n cos(i),  n = sqrt(mu/a^3), p = a(1-e^2)   [rad/s]
    i = np.radians(i_deg)
    n = np.sqrt(mu_km3s2 / (a_km**3))
    p = a_km * (1.0 - e**2)
    return -1.5 * J2 * (Re_km/p)**2 * n * np.cos(i)  # rad/s

def measure_raan_slope_deg_per_day(t_s, Om_deg):
    # declive por ajuste linear da série desenrolada
    Om_unw = _unwrap_deg(Om_deg)
    m_deg_per_s = np.polyfit(t_s, Om_unw, 1)[0]
    return 86400.0 * m_deg_per_s  # deg/day

def summarize_sat(name, t, a_ser, e_ser, i_ser, Om_on_deg):
    a_bar = float(np.nanmean(a_ser))
    e_bar = float(np.nanmean(e_ser))
    i_bar = float(np.nanmean(i_ser))
    dur_d = float((t[-1] - t[0]) / 86400.0)

    raan_th_deg_day = np.degrees(j2_raan_rate_theory(a_bar, e_bar, i_bar, MU, earth_radius, P.J2)) * 86400.0
    raan_num_deg_day = measure_raan_slope_deg_per_day(t, Om_on_deg)
    err_rel_pct = 100.0 * (raan_num_deg_day - raan_th_deg_day) / abs(raan_th_deg_day if raan_th_deg_day!=0 else 1.0)

    return dict(name=name, a=a_bar, e=e_bar, i=i_bar,
                th=raan_th_deg_day, num=raan_num_deg_day, err=err_rel_pct, dur=dur_d)

rows = [
    summarize_sat("VH Up",   t_up,   a_up_on,  e_up_on,  I_up_on,  Om_up_on),
    summarize_sat("V Only",  t_v,    a_v_on,   e_v_on,   I_v_on,   Om_v_on),
    summarize_sat("VH Down", t_down, a_dn_on,  e_dn_on,  I_dn_on,  Om_dn_on),
]

# --- monta a tabela LaTeX ---
table_tex = []
table_tex.append(r"\begin{table}[ht]")
table_tex.append(r"\centering")
table_tex.append(r"\caption{Comparação entre a precessão nodal teórica por $J_2$ e a medida numericamente a partir de $\Omega(t)$.}")
table_tex.append(r"\label{tab:raan_j2_comp}")
table_tex.append(r"\sisetup{table-number-alignment = center, round-mode = places, round-precision = 6}")
table_tex.append(r"\begin{tabular}{l S[table-format=5.0] S[table-format=1.5] S[table-format=3.3] S[table-format=1.6] S[table-format=1.6] S[table-format=+2.2] S[table-format=2.1]}")
table_tex.append(r"\toprule")
table_tex.append(r"Satélite & {$\bar{a}$\,[km]} & {$\bar{e}$} & {$\bar{i}$\,[deg]} & {$\dot{\Omega}_{\text{teo.}}$\,[deg/d]} & {$\dot{\Omega}_{\text{num.}}$\,[deg/d]} & {$\varepsilon$\,[\%]} & {Duração\,[d]} \\")
table_tex.append(r"\midrule")
for r in rows:
    table_tex.append(
        fr"{r['name']} & {r['a']:.0f} & {r['e']:.5f} & {r['i']:.3f} & {r['th']:.6f} & {r['num']:.6f} & {r['err']:.2f} & {r['dur']:.1f} \\"
    )
table_tex.append(r"\bottomrule")
table_tex.append(r"\end{tabular}")
table_tex.append(r"\end{table}")

table_tex = "\n".join(table_tex)
print("\n" + table_tex)

# grava em arquivo para usar com \input{...} no Overleaf
with open("raan_comparison_table.tex", "w", encoding="utf-8") as f:
    f.write(table_tex)
print("\n[Tabela salva em 'raan_comparison_table.tex']")


# --------- Paleta
COL_UP   = "#1f77b4"  # azul
COL_V    = "#2ca02c"  # verde
COL_DOWN = "#d62728"  # vermelho

# --------- GRÁFICOS INDIVIDUAIS (por satélite): 3D e i(ν) COM vs SEM J2
_plot_satellite_3d_comparison("VH Up",   X_up,   X_up_off,  RE=earth_radius, color=COL_UP)
_plot_satellite_i_vs_nu_comparison("VH Up",   Nu_up_on,  I_up_on,  Nu_up_off,  I_up_off,  color=COL_UP)

_plot_satellite_3d_comparison("V Only",  X_v,    X_v_off,  RE=earth_radius, color=COL_V)
_plot_satellite_i_vs_nu_comparison("V Only",  Nu_v_on,   I_v_on,   Nu_v_off,   I_v_off,   color=COL_V)

_plot_satellite_3d_comparison("VH Down", X_down, X_dn_off, RE=earth_radius, color=COL_DOWN)
_plot_satellite_i_vs_nu_comparison("VH Down", Nu_dn_on,  I_dn_on,  Nu_dn_off,  I_dn_off,  color=COL_DOWN)

# --------- GRÁFICO COMBINADO 3D: COM (linha) × SEM (tracejada) para os 3 satélites
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
u, vgrid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = earth_radius * np.cos(u)*np.sin(vgrid)
y_e = earth_radius * np.sin(u)*np.sin(vgrid)
z_e = earth_radius * np.cos(vgrid)
ax.plot_wireframe(x_e, y_e, z_e, color="g", alpha=0.25)
ax.set_box_aspect([1,1,1])

# ON (linha cheia)
ax.plot3D(X_up[0,:],   X_up[1,:],   X_up[2,:],   '-',  color=COL_UP,   lw=1.6, label="VH Up  (J2 ON)")
ax.plot3D(X_v[0,:],    X_v[1,:],    X_v[2,:],    '-',  color=COL_V,    lw=1.6, label="V Only (J2 ON)")
ax.plot3D(X_down[0,:], X_down[1,:], X_down[2,:], '-',  color=COL_DOWN, lw=1.6, label="VH Down (J2 ON)")

# OFF (tracejada)
ax.plot3D(X_up_off[0,:],  X_up_off[1,:],  X_up_off[2,:],  '--', color=COL_UP,   lw=1.6, label="VH Up  (J2 OFF)")
ax.plot3D(X_v_off[0,:],   X_v_off[1,:],   X_v_off[2,:],   '--', color=COL_V,    lw=1.6, label="V Only (J2 OFF)")
ax.plot3D(X_dn_off[0,:],  X_dn_off[1,:],  X_dn_off[2,:],  '--', color=COL_DOWN, lw=1.6, label="VH Down (J2 OFF)")

ax.set_title("Constelação — Trajetória 3D: COM (cheia) × SEM J2 (tracejada)")
ax.legend()
ax.axis('equal')
plt.show()

# --------- RAAN(t): deriva secular por J2 (desenrolado)
plt.figure()
plt.plot(t_up,   _unwrap_deg(Om_up_on),  '-',  color=COL_UP,   lw=1.2, label="VH Up  (J2 ON)")
plt.plot(t_up,   _unwrap_deg(Om_up_off), '--', color=COL_UP,   lw=1.2, label="VH Up  (J2 OFF)")
plt.plot(t_v,    _unwrap_deg(Om_v_on),   '-',  color=COL_V,    lw=1.2, label="V Only (J2 ON)")
plt.plot(t_v,    _unwrap_deg(Om_v_off),  '--', color=COL_V,    lw=1.2, label="V Only (J2 OFF)")
plt.plot(t_down, _unwrap_deg(Om_dn_on),  '-',  color=COL_DOWN, lw=1.2, label="VH Down (J2 ON)")
plt.plot(t_down, _unwrap_deg(Om_dn_off), '--', color=COL_DOWN, lw=1.2, label="VH Down (J2 OFF)")
plt.xlabel("Tempo [s]"); plt.ylabel("Ω (RAAN) [graus] (desenrolado)")
plt.title("Deriva secular de Ω (efeito J2)")
plt.grid(alpha=0.3); plt.legend(); plt.show()

# --------- Inclinação i(t): COM x SEM J2
plt.figure()
plt.plot(t_up,   I_up_on,  '-',  color=COL_UP,   lw=1.2, label="VH Up  (J2 ON)")
plt.plot(t_up,   I_up_off, '--', color=COL_UP,   lw=1.2, label="VH Up  (J2 OFF)")
plt.plot(t_v,    I_v_on,   '-',  color=COL_V,    lw=1.2, label="V Only (J2 ON)")
plt.plot(t_v,    I_v_off,  '--', color=COL_V,    lw=1.2, label="V Only (J2 OFF)")
plt.plot(t_down, I_dn_on,  '-',  color=COL_DOWN, lw=1.2, label="VH Down (J2 ON)")
plt.plot(t_down, I_dn_off, '--', color=COL_DOWN, lw=1.2, label="VH Down (J2 OFF)")
plt.xlabel("Tempo [s]"); plt.ylabel("Inclinação i [graus]")
plt.title("Inclinação i(t): COM x SEM J2")
plt.grid(alpha=0.3); plt.legend(); plt.show()

# --------- i(ν): COM (opaco) × SEM J2 (transparente)
fig2, ax2 = plt.subplots()
_plot_i_vs_nu_segmentado(Nu_up_on,  I_up_on,  ax=ax2, color=COL_UP,   lw=1.2, label="VH Up  (J2 ON)")
_plot_i_vs_nu_segmentado(Nu_up_off, I_up_off, ax=ax2, color=COL_UP,   lw=1.2, label="VH Up  (J2 OFF)", alpha=0.35)
_plot_i_vs_nu_segmentado(Nu_v_on,   I_v_on,   ax=ax2, color=COL_V,    lw=1.2, label="V Only (J2 ON)")
_plot_i_vs_nu_segmentado(Nu_v_off,  I_v_off,  ax=ax2, color=COL_V,    lw=1.2, label="V Only (J2 OFF)", alpha=0.35)
_plot_i_vs_nu_segmentado(Nu_dn_on,  I_dn_on,  ax=ax2, color=COL_DOWN, lw=1.2, label="VH Down (J2 ON)")
_plot_i_vs_nu_segmentado(Nu_dn_off, I_dn_off, ax=ax2, color=COL_DOWN, lw=1.2, label="VH Down (J2 OFF)", alpha=0.35)
ax2.set_title("Inclinação i(ν): COM (opaco) × SEM J2 (tracejada/transparente)")
ax2.legend(markerscale=4, fontsize=9)
plt.show()

# --------- Diferenças (ON – OFF): ΔΩ e Δi ao longo do tempo
plt.figure()
plt.plot(t_up,   _unwrap_deg(Om_up_on)  - _unwrap_deg(Om_up_off),  '-', color=COL_UP,   lw=1.2, label="ΔΩ VH Up")
plt.plot(t_v,    _unwrap_deg(Om_v_on)   - _unwrap_deg(Om_v_off),   '-', color=COL_V,    lw=1.2, label="ΔΩ V Only")
plt.plot(t_down, _unwrap_deg(Om_dn_on)  - _unwrap_deg(Om_dn_off),  '-', color=COL_DOWN, lw=1.2, label="ΔΩ VH Down")
plt.xlabel("Tempo [s]"); plt.ylabel("ΔΩ [graus]")
plt.title("Diferença de RAAN (J2 ON – J2 OFF)")
plt.grid(alpha=0.3); plt.legend(); plt.show()

plt.figure()
plt.plot(t_up,   I_up_on - I_up_off,   '-', color=COL_UP,   lw=1.2, label="Δi VH Up")
plt.plot(t_v,    I_v_on  - I_v_off,    '-', color=COL_V,    lw=1.2, label="Δi V Only")
plt.plot(t_down, I_dn_on - I_dn_off,   '-', color=COL_DOWN, lw=1.2, label="Δi VH Down")
plt.xlabel("Tempo [s]"); plt.ylabel("Δi [graus]")
plt.title("Diferença de Inclinação (J2 ON – J2 OFF)")
plt.grid(alpha=0.3); plt.legend(); plt.show()
