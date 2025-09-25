# main_constelacao.py
# Execução consolidada das três simulações (VH UP, V ONLY, VH DOWN)
# Agora com chaves de perturbação explícitas: j2: bool e drag: bool
# - baseline: j2=True, drag=True
# - refs:     J2 OFF (drag ON) e DRAG OFF (J2 ON)
# Mantém diagnósticos e gráficos J2; adiciona gráficos de arrasto (a(t), e(t))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def _call_sim_with_flags(module, **flags):
    """
    Tenta module.simulate(**flags).
    Se não aceitar, tenta compat de versões antigas (apenas j2=...).
    Se ainda assim não rolar, retorna None.
    """
    try:
        return module.simulate(**flags)
    except TypeError:
        pass
    try:
        return module.simulate(j2=flags.get("j2", True))
    except TypeError:
        pass
    return None

def _two_body_propagate(x0, t, mu):
    """
    Fallback SEM PERTURBAÇÃO: 2-corpos (sem J2/drag/empuxo). Mantém massa constante se existir.
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
    out = np.empty((arr_src.shape[0], t_ref.size), dtype=arr_src.dtype)
    for i in range(arr_src.shape[0]):
        out[i, :] = interp(t_ref, t_src, arr_src[i, :])
    return out

def _plot_satellite_3d_comparison(name, X_on, X_off, RE, color="#1f77b4", tag_on="ON", tag_off="OFF"):
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
    ax.plot3D(X_on[0,:],  X_on[1,:],  X_on[2,:],  '-',  color=color, lw=1.6, label=f"{name} ({tag_on})")
    ax.plot3D(X_off[0,:], X_off[1,:], X_off[2,:], '--', color=color, lw=1.6, label=f"{name} ({tag_off})")
    ax.set_title(f"{name}: Trajetória 3D — {tag_on} (linha cheia) × {tag_off} (tracejada)")
    ax.legend()
    ax.axis('equal')
    plt.show()

def _plot_satellite_i_vs_nu_comparison(name, Nu_on, I_on, Nu_off, I_off, color="#1f77b4", tag_on="ON", tag_off="OFF"):
    fig, ax = plt.subplots()
    _plot_i_vs_nu_segmentado(Nu_on,  I_on,  ax=ax, color=color,      lw=1.6, label=f"{name} ({tag_on})")
    _plot_i_vs_nu_segmentado(Nu_off, I_off, ax=ax, color=color,      lw=1.6, label=f"{name} ({tag_off})", alpha=0.35)
    ax.set_title(f"{name}: Inclinação i(ν) — {tag_on} × {tag_off}")
    ax.legend()
    plt.show()

# ----------------- Parâmetros Terra -----------------
P = EarthParams()  # (mu, Re, J2)
earth_radius = getattr(P, "Re", 6378.0)
MU = getattr(P, "mu", 3.986e5)

# ================= Simulações baseline (J2 ON, DRAG ON) =================
FLAGS_ON = dict(j2=True, drag=True)

print("Simulando satélite VH UP (J2 ON, DRAG ON)...")
res_up = _call_sim_with_flags(sat_vh_up, **FLAGS_ON) or sat_vh_up.simulate()
print("Simulando satélite V ONLY (J2 ON, DRAG ON)...")
res_v = _call_sim_with_flags(sat_v_only, **FLAGS_ON) or sat_v_only.simulate()
print("Simulando satélite VH DOWN (J2 ON, DRAG ON)...")
res_down = _call_sim_with_flags(sat_vh_down, **FLAGS_ON) or sat_vh_down.simulate()

t_up, X_up, nus_up, incs_up, elems_up = res_up
t_v,  X_v,  nus_v,  incs_v,  elems_v  = res_v
t_dn, X_dn, nus_dn, incs_dn, elems_dn = res_down

# ---------- Diagnósticos (J2 ON, DRAG ON) ----------
print("\n##### Diagnósticos de elementos (efeitos compatíveis com J2) #####")
diag_up   = diagnose_orbital_changes(t_up,   X_up,   params=P, label="VH UP")
diag_v    = diagnose_orbital_changes(t_v,    X_v,    params=P, label="V ONLY")
diag_down = diagnose_orbital_changes(t_dn, X_dn, params=P, label="VH DOWN")

# ================= RAAN(t) dos 3 + janelas de empuxo H =================
# --- helpers de janela (mesmos parâmetros usados nos módulos) ---
THRUST_INTERVAL_DEG = 30.0
MEAN_THETA_LIST_DEG = [180.0]   # centrado no apogeu

def wrap_deg(a):  return np.remainder(a, 360.0)

def angle_in_window_deg(theta_deg, center_deg, width_deg):
    half = 0.5 * width_deg
    lo = wrap_deg(center_deg - half)
    hi = wrap_deg(center_deg + half)
    th = wrap_deg(theta_deg)
    return (th >= lo) & (th <= hi) if lo <= hi else ((th >= lo) | (th <= hi))

def in_any_window(theta_deg):
    return any(angle_in_window_deg(theta_deg, c, THRUST_INTERVAL_DEG)
               for c in MEAN_THETA_LIST_DEG)

def _mask_to_spans(t, mask_bool):
    t = np.asarray(t, float); mask = np.asarray(mask_bool, bool)
    if mask.size == 0: return []
    rises = np.where(np.diff(mask.astype(np.int8)) == +1)[0] + 1
    falls = np.where(np.diff(mask.astype(np.int8)) == -1)[0] + 1
    if mask[0]:  rises = np.r_[0, rises]
    if mask[-1]: falls = np.r_[falls, mask.size-1]
    return list(zip(t[rises], t[falls]))

def add_thrust_spans(ax, t, mask_bool, *, color, alpha=0.15, label=None):
    spans = _mask_to_spans(t, mask_bool)
    for t0, t1 in spans:
        ax.axvspan(t0, t1, color=color, alpha=alpha, lw=0)
    if label:
        handles, labels = ax.get_legend_handles_labels()
        patch = mpatches.Patch(color=color, alpha=alpha, label=label)
        if label not in labels:
            ax.legend(handles + [patch], labels + [label])

# --- máscaras de empuxo H ligado (considera janela em ν e propelente disponível) ---
m_dry = 15.0  # kg (ajuste se diferente nos módulos)
mask_up   = (X_up[6, :] > m_dry) & np.array([in_any_window(nu)   for nu in nus_up],   bool)
mask_dn   = (X_dn[6, :] > m_dry) & np.array([in_any_window(nu)   for nu in nus_dn],   bool)
mask_v    = np.zeros_like(t_v, dtype=bool)  # V Only não tem H

# --- RAAN desenrolado de cada satélite ---
def _raan_series(t, X, elems, mu=MU):
    if elems is not None and len(elems) == X.shape[1]:
        Om = np.array([e.ascending_node for e in elems], float)
    else:
        Om = np.empty(X.shape[1], float)
        for k in range(X.shape[1]):
            Om[k] = get_orbital_elements(X[0:3, k], X[3:6, k], mu).ascending_node
    return _unwrap_deg(Om)

Om_up_unw = _raan_series(t_up,   X_up,   elems_up)
Om_v_unw  = _raan_series(t_v,    X_v,    elems_v)
Om_dn_unw = _raan_series(t_dn,   X_dn,   elems_dn)

# --- plot único RAAN (baseline) ---
COL_UP, COL_V, COL_DN = "#1f77b4", "#2ca02c", "#d62728"
t_up_d, t_v_d, t_dn_d = t_up/86400.0, t_v/86400.0, t_dn/86400.0

fig, ax = plt.subplots()
ax.plot(t_up_d, Om_up_unw,   '-', color=COL_UP, lw=1.6, label="VH Up  — Ω (J2+DRAG)")
ax.plot(t_v_d,  Om_v_unw,    '-', color=COL_V,  lw=1.6, label="V Only — Ω (J2+DRAG)")
ax.plot(t_dn_d, Om_dn_unw,   '-', color=COL_DN, lw=1.6, label="VH Down — Ω (J2+DRAG)")
add_thrust_spans(ax, t_up_d,  mask_up,   color=COL_UP,  alpha=0.15, label="H ON (VH Up)")
add_thrust_spans(ax, t_dn_d,  mask_dn,   color=COL_DN,  alpha=0.15, label="H ON (VH Down)")
ax.set_xlabel("Tempo [dias]")
ax.set_ylabel("RAAN, Ω [graus] (desenrolado)")
ax.set_title("RAAN dos 3 satélites (baseline J2+DRAG) com janelas de empuxo H")
ax.grid(True); ax.legend(); plt.show()

# ----------------- Gráfico 3D (baseline) -----------------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
u, vgrid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = earth_radius * np.cos(u)*np.sin(vgrid)
y_e = earth_radius * np.sin(u)*np.sin(vgrid)
z_e = earth_radius * np.cos(vgrid)
ax.plot_wireframe(x_e, y_e, z_e, color="g", alpha=0.3)
ax.set_box_aspect([1, 1, 1])
ax.plot3D(X_up[0,:],   X_up[1,:],   X_up[2,:],   'b-', label="VH Up (J2+DRAG)")
ax.plot3D(X_v[0,:],    X_v[1,:],    X_v[2,:],    'g-', label="V Only (J2+DRAG)")
ax.plot3D(X_dn[0,:],   X_dn[1,:],   X_dn[2,:],   'r-', label="VH Down (J2+DRAG)")
ax.set_title("Constelação - 3 Satélites (J2+DRAG)")
ax.legend(); ax.axis('equal'); plt.show()

# ----------------- i(ν) (baseline) -----------------
plt.figure()
plt.plot(nus_up,   incs_up,   'b.', ms=0.7, label="VH Up")
plt.plot(nus_v,    incs_v,    'g.', ms=0.7, label="V Only")
plt.plot(nus_dn,   incs_dn,   'r.', ms=0.7, label="VH Down")
plt.xlim(0, 360)
plt.xlabel("Anomalia Verdadeira ν [graus]")
plt.ylabel("Inclinação i [graus]")
plt.title("Inclinação vs ν - (J2+DRAG)")
plt.legend(); plt.grid(alpha=0.3); plt.show()

# ----------------- Massa vs tempo (baseline) -----------------
plt.figure()
plt.plot(t_up,   X_up[6, :],   'b-', label="VH Up")
plt.plot(t_v,    X_v[6, :],    'g-', label="V Only")
plt.plot(t_dn,   X_dn[6, :],   'r-', label="VH Down")
plt.xlabel("Tempo [s]")
plt.ylabel("Massa [kg]")
plt.title("Consumo de Propelente (J2+DRAG)")
plt.legend(); plt.grid(alpha=0.3); plt.show()

# =====================================================================
# ADENDO 1: COM x SEM J2 (drag mantido ON) — (efeito J2)
# =====================================================================
print("\n##### Adendo: Comparação COM x SEM J2 (drag ON) #####")

def _sim_off_j2(module, t_on, X_on):
    res_off = _call_sim_with_flags(module, j2=False, drag=True)
    if res_off is None:
        # fallback: 2-corpos
        X_off  = _two_body_propagate(X_on[:, 0].copy(), t_on, MU)
        Nu, I, Om, W, e, a = _elements_series(t_on, X_off, MU)
        return t_on, X_off, Nu, I, (Om, W, e, a)
    t_off, X_off, Nu, I, elems = res_off
    if not np.allclose(t_on, t_off):
        X_off  = _sync_series(t_on, t_off, X_off)
        Nu     = _sync_series(t_on, t_off, Nu)
        I      = _sync_series(t_on, t_off, I)
    _, _, Om, W, e, a = _elements_series(t_on, X_off, MU)
    return t_on, X_off, Nu, I, (Om, W, e, a)

t_up_offJ2,  X_up_offJ2,  Nu_up_offJ2,  I_up_offJ2,  (Om_up_offJ2,  W_up_offJ2,  e_up_offJ2,  a_up_offJ2)  = _sim_off_j2(sat_vh_up,   t_up,  X_up)
t_v_offJ2,   X_v_offJ2,   Nu_v_offJ2,   I_v_offJ2,   (Om_v_offJ2,   W_v_offJ2,   e_v_offJ2,   a_v_offJ2)   = _sim_off_j2(sat_v_only, t_v,   X_v)
t_dn_offJ2,  X_dn_offJ2,  Nu_dn_offJ2,  I_dn_offJ2,  (Om_dn_offJ2,  W_dn_offJ2,  e_dn_offJ2,  a_dn_offJ2)  = _sim_off_j2(sat_vh_down, t_dn,  X_dn)

# séries baseline para esses gráficos
Nu_up_on,  I_up_on,  Om_up_on,  W_up_on,  e_up_on,  a_up_on  = _elements_series(t_up,  X_up,  MU)
Nu_v_on,   I_v_on,   Om_v_on,   W_v_on,   e_v_on,   a_v_on   = _elements_series(t_v,   X_v,   MU)
Nu_dn_on,  I_dn_on,  Om_dn_on,  W_dn_on,  e_dn_on,  a_dn_on  = _elements_series(t_dn,  X_dn,  MU)

# --------- RAAN(t): deriva secular por J2 (desenrolado)
plt.figure()
plt.plot(t_up,  _unwrap_deg(Om_up_on),   '-',  color=COL_UP,   lw=1.2, label="VH Up  (J2 ON)")
plt.plot(t_up,  _unwrap_deg(Om_up_offJ2),'--', color=COL_UP,   lw=1.2, label="VH Up  (J2 OFF)")
plt.plot(t_v,   _unwrap_deg(Om_v_on),    '-',  color=COL_V,    lw=1.2, label="V Only (J2 ON)")
plt.plot(t_v,   _unwrap_deg(Om_v_offJ2), '--', color=COL_V,    lw=1.2, label="V Only (J2 OFF)")
plt.plot(t_dn,  _unwrap_deg(Om_dn_on),   '-',  color=COL_DN,   lw=1.2, label="VH Down (J2 ON)")
plt.plot(t_dn,  _unwrap_deg(Om_dn_offJ2),'--', color=COL_DN,   lw=1.2, label="VH Down (J2 OFF)")
plt.xlabel("Tempo [s]"); plt.ylabel("Ω (RAAN) [graus] (desenrolado)")
plt.title("Deriva secular de Ω — efeito de J2 (drag ON)")
plt.grid(alpha=0.3); plt.legend(); plt.show()

# --------- Inclinação i(t): COM x SEM J2
plt.figure()
plt.plot(t_up,  I_up_on,   '-',  color=COL_UP,   lw=1.2, label="VH Up  (J2 ON)")
plt.plot(t_up,  I_up_offJ2,'--', color=COL_UP,   lw=1.2, label="VH Up  (J2 OFF)")
plt.plot(t_v,   I_v_on,    '-',  color=COL_V,    lw=1.2, label="V Only (J2 ON)")
plt.plot(t_v,   I_v_offJ2, '--', color=COL_V,    lw=1.2, label="V Only (J2 OFF)")
plt.plot(t_dn,  I_dn_on,   '-',  color=COL_DN,   lw=1.2, label="VH Down (J2 ON)")
plt.plot(t_dn,  I_dn_offJ2,'--', color=COL_DN,   lw=1.2, label="VH Down (J2 OFF)")
plt.xlabel("Tempo [s]"); plt.ylabel("Inclinação i [graus]")
plt.title("Inclinação i(t): J2 ON × J2 OFF (drag ON)")
plt.grid(alpha=0.3); plt.legend(); plt.show()

# =====================================================================
# ADENDO 2: COM x SEM DRAG (J2 mantido ON) — (efeito arrasto)
# =====================================================================
print("\n##### Adendo: Comparação COM x SEM DRAG (J2 ON) #####")

def _sim_off_drag(module, t_on, X_on):
    res_off = _call_sim_with_flags(module, j2=True, drag=False)
    if res_off is None:
        # fallback: 2-corpos
        X_off  = _two_body_propagate(X_on[:, 0].copy(), t_on, MU)
        Nu, I, Om, W, e, a = _elements_series(t_on, X_off, MU)
        return t_on, X_off, Nu, I, (Om, W, e, a)
    t_off, X_off, Nu, I, elems = res_off
    if not np.allclose(t_on, t_off):
        X_off  = _sync_series(t_on, t_off, X_off)
        Nu     = _sync_series(t_on, t_off, Nu)
        I      = _sync_series(t_on, t_off, I)
    _, _, Om, W, e, a = _elements_series(t_on, X_off, MU)
    return t_on, X_off, Nu, I, (Om, W, e, a)

t_up_offDR,  X_up_offDR,  Nu_up_offDR,  I_up_offDR,  (Om_up_offDR,  W_up_offDR,  e_up_offDR,  a_up_offDR)  = _sim_off_drag(sat_vh_up,   t_up,  X_up)
t_v_offDR,   X_v_offDR,   Nu_v_offDR,   I_v_offDR,   (Om_v_offDR,   W_v_offDR,   e_v_offDR,   a_v_offDR)   = _sim_off_drag(sat_v_only, t_v,   X_v)
t_dn_offDR,  X_dn_offDR,  Nu_dn_offDR,  I_dn_offDR,  (Om_dn_offDR,  W_dn_offDR,  e_dn_offDR,  a_dn_offDR)  = _sim_off_drag(sat_vh_down, t_dn,  X_dn)

# ---- Semieixo maior a(t): DRAG ON × DRAG OFF (J2 ON)
plt.figure()
plt.plot(t_up/86400.0, a_up_on,     '-',  color=COL_UP,   lw=1.3, label="VH Up (DRAG ON)")
plt.plot(t_up/86400.0, a_up_offDR,  '--', color=COL_UP,   lw=1.3, label="VH Up (DRAG OFF)")
plt.plot(t_v/86400.0,  a_v_on,      '-',  color=COL_V,    lw=1.3, label="V Only (DRAG ON)")
plt.plot(t_v/86400.0,  a_v_offDR,   '--', color=COL_V,    lw=1.3, label="V Only (DRAG OFF)")
plt.plot(t_dn/86400.0, a_dn_on,     '-',  color=COL_DN,   lw=1.3, label="VH Down (DRAG ON)")
plt.plot(t_dn/86400.0, a_dn_offDR,  '--', color=COL_DN,   lw=1.3, label="VH Down (DRAG OFF)")
plt.xlabel("Tempo [dias]"); plt.ylabel("a [km]")
plt.title("Semieixo maior a(t): efeito do arrasto (J2 ON)")
plt.grid(alpha=0.3); plt.legend(); plt.show()

# ---- Excentricidade e(t): DRAG ON × DRAG OFF (J2 ON)
plt.figure()
plt.plot(t_up/86400.0, e_up_on,     '-',  color=COL_UP,   lw=1.3, label="VH Up (DRAG ON)")
plt.plot(t_up/86400.0, e_up_offDR,  '--', color=COL_UP,   lw=1.3, label="VH Up (DRAG OFF)")
plt.plot(t_v/86400.0,  e_v_on,      '-',  color=COL_V,    lw=1.3, label="V Only (DRAG ON)")
plt.plot(t_v/86400.0,  e_v_offDR,   '--', color=COL_V,    lw=1.3, label="V Only (DRAG OFF)")
plt.plot(t_dn/86400.0, e_dn_on,     '-',  color=COL_DN,   lw=1.3, label="VH Down (DRAG ON)")
plt.plot(t_dn/86400.0, e_dn_offDR,  '--', color=COL_DN,   lw=1.3, label="VH Down (DRAG OFF)")
plt.xlabel("Tempo [dias]"); plt.ylabel("e [-]")
plt.title("Excentricidade e(t): efeito do arrasto (J2 ON)")
plt.grid(alpha=0.3); plt.legend(); plt.show()

# ---- Comparações 3D e i(ν) para DRAG (opcional mas útil)
_plot_satellite_3d_comparison("VH Up",   X_up,   X_up_offDR,  RE=earth_radius, color=COL_UP,   tag_on="DRAG ON", tag_off="DRAG OFF")
_plot_satellite_i_vs_nu_comparison("VH Up",   Nu_up_on,  I_up_on,  Nu_up_offDR,  I_up_offDR,  color=COL_UP,   tag_on="DRAG ON", tag_off="DRAG OFF")

_plot_satellite_3d_comparison("V Only", X_v,    X_v_offDR,  RE=earth_radius, color=COL_V,    tag_on="DRAG ON", tag_off="DRAG OFF")
_plot_satellite_i_vs_nu_comparison("V Only",  Nu_v_on,   I_v_on,   Nu_v_offDR,   I_v_offDR,   color=COL_V,    tag_on="DRAG ON", tag_off="DRAG OFF")

_plot_satellite_3d_comparison("VH Down",X_dn,   X_dn_offDR, RE=earth_radius, color=COL_DN,   tag_on="DRAG ON", tag_off="DRAG OFF")
_plot_satellite_i_vs_nu_comparison("VH Down", Nu_dn_on,  I_dn_on,  Nu_dn_offDR,  I_dn_offDR,  color=COL_DN,   tag_on="DRAG ON", tag_off="DRAG OFF")

# =================== TABELA: Ω̇(J2 teórico) × Ω̇(numérico) ===================
def j2_raan_rate_theory(a_km, e, i_deg, mu_km3s2, Re_km, J2):
    # dΩ/dt = -(3/2) J2 (Re/p)^2 n cos(i),  n = sqrt(mu/a^3), p = a(1-e^2)   [rad/s]
    i = np.radians(i_deg)
    n = np.sqrt(mu_km3s2 / (a_km**3))
    p = a_km * (1.0 - e**2)
    return -1.5 * J2 * (Re_km/p)**2 * n * np.cos(i)  # rad/s

def measure_raan_slope_deg_per_day(t_s, Om_deg):
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
    err_rel_pct = 100.0 * (raan_num_deg_day - raan_th_deg_day) / (abs(raan_th_deg_day) if raan_th_deg_day!=0 else 1.0)
    return dict(name=name, a=a_bar, e=e_bar, i=i_bar,
                th=raan_th_deg_day, num=raan_num_deg_day, err=err_rel_pct, dur=dur_d)

rows = [
    summarize_sat("VH Up",   t_up,   a_up_on,  e_up_on,  I_up_on,  Om_up_on),
    summarize_sat("V Only",  t_v,    a_v_on,   e_v_on,   I_v_on,   Om_v_on),
    summarize_sat("VH Down", t_dn,   a_dn_on,  e_dn_on,  I_dn_on,  Om_dn_on),
]

# --- monta a tabela LaTeX ---
table_tex = []
table_tex.append(r"\begin{table}[ht]")
table_tex.append(r"\centering")
table_tex.append(r"\caption{Comparação entre a precessão nodal teórica por $J_2$ e a medida numericamente a partir de $\Omega(t)$ (drag ON).}")
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

with open("raan_comparison_table.tex", "w", encoding="utf-8") as f:
    f.write(table_tex)
print("\n[Tabela salva em 'raan_comparison_table.tex']")

# --------- Paleta (para funções que usam defaults) ---------
COL_UP   = "#1f77b4"  # azul
COL_V    = "#2ca02c"  # verde
COL_DN   = "#d62728"  # vermelho
