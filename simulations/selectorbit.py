# main_constelacao_last_orbit.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import inspect

# Módulos dos satélites (cada um precisa expor simulate())
import sat_vh_down as S_DOWN
import sat_vh_up   as S_UP
import sat_v_only  as S_V

# Utils do projeto
from utils.visualization import ElementsSeries, plot_classic_orbital_elements
from utils.orbital_elements import *
from utils.orbitalElementsOperations import *

# ------------------------------- Constantes -------------------------------
EARTH_RADIUS_KM = 6378.0
MU = 3.986e5  # km^3/s^2

# ------------------------------- Helpers ---------------------------------
def _as_7xN(X):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X deve ser 2D.")
    if X.shape[0] in (6, 7) and X.shape[1] not in (6, 7):
        X7 = X
    elif X.shape[1] in (6, 7) and X.shape[0] not in (6, 7):
        X7 = X.T
    else:
        X7 = X
    if X7.shape[0] == 6:  # sem massa
        X7 = np.vstack([X7, np.full((1, X7.shape[1]), np.nan)])
    return X7

def _unwrap_deg(a_deg):
    return np.degrees(np.unwrap(np.radians(np.asarray(a_deg, float))))

def _phase_deg_and_incl_from_states(X):
    """Fase robusta (ν ou u) e inclinação, em graus."""
    N = X.shape[1]
    phase = np.empty(N); incs = np.empty(N)
    for k in range(N):
        r = X[0:3, k]; v = X[3:6, k]
        e_now = get_eccentricity(r, v, MU)
        phase[k] = get_argument_of_latitude(r, v, MU) if e_now < 1e-5 else get_true_anomaly(r, v, MU)
        incs[k]  = get_inclination(r, v, MU)
    return phase, incs

def _elements_from_states(X):
    N = X.shape[1]
    a      = np.empty(N); e      = np.empty(N); i_deg  = np.empty(N)
    Om_deg = np.empty(N); w_deg  = np.empty(N); nu_deg = np.empty(N)
    u_deg  = np.empty(N); ltrue  = np.empty(N); energy = np.empty(N)
    for k in range(N):
        r = X[0:3, k]; v = X[3:6, k]
        a[k]       = get_major_axis(r, v, MU)
        e[k]       = get_eccentricity(r, v, MU)
        i_deg[k]   = get_inclination(r, v, MU)
        Om_deg[k]  = get_ascending_node(r, v, MU)
        w_deg[k]   = get_argument_of_perigee(r, v, MU)
        nu_deg[k]  = get_true_anomaly(r, v, MU)
        u_deg[k]   = get_argument_of_latitude(r, v, MU)
        ltrue[k]   = get_true_longitude(r, v, MU)
        try:
            energy[k] = get_specific_energy(r, v, MU)
        except Exception:
            energy[k] = 0.5*np.dot(v, v) - MU/np.linalg.norm(r)
    return ElementsSeries(a, e, i_deg, Om_deg, w_deg, nu_deg, u_deg, ltrue, energy)

def _plot_earth_sphere(ax):
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    x_e = EARTH_RADIUS_KM * np.cos(u) * np.sin(v)
    y_e = EARTH_RADIUS_KM * np.sin(u) * np.sin(v)
    z_e = EARTH_RADIUS_KM * np.cos(v)
    ax.plot_wireframe(x_e, y_e, z_e, color="g", alpha=0.25, linewidth=0.6)

def _force_equal_3d_limits(ax, X_list):
    rmax = EARTH_RADIUS_KM
    for X in X_list:
        rmax = max(rmax, float(np.linalg.norm(X[0:3, :], axis=0).max()))
    R = 1.05 * rmax
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
    ax.set_box_aspect([1, 1, 1])

def _plot_i_vs_phase_segmentado(phi_deg, inc_deg, *, ax=None, color=None, label=None, **kw):
    phi_deg = np.asarray(phi_deg, float); inc_deg = np.asarray(inc_deg, float)
    if ax is None: fig, ax = plt.subplots()
    dn = np.diff(phi_deg); wraps = np.where(dn < -180.0)[0]
    start = 0; first = True
    for w in wraps:
        ax.plot(phi_deg[start:w+1], inc_deg[start:w+1], color=color, label=(label if first else None), **kw)
        first = False; start = w + 1
    ax.plot(phi_deg[start:], inc_deg[start:], color=color, label=(label if first else None), **kw)
    ax.set_xlabel(r'Fase (ν ou $u$) [deg]'); ax.set_ylabel(r'$i$ [deg]')
    ax.set_xlim(0, 360); ax.grid(True); return ax

def _try_disable_perturbations_in_module(mod):
    for attr in ("_USE_J2", "_USE_J22", "_DRAG_ON", "_J2_ON", "_J22_ON", "J2_ON", "J22_ON", "DRAG_ON"):
        if hasattr(mod, attr):
            try: setattr(mod, attr, False)
            except Exception: pass

def _simulate_no_perturbations(mod):
    """Chama mod.simulate() sem perturbações e normaliza a saída."""
    _try_disable_perturbations_in_module(mod)
    sim = mod.simulate
    try:
        sig = inspect.signature(sim); params = sig.parameters
        kwargs = {}
        for k in ("j2", "j22", "drag"):
            if k in params: kwargs[k] = False
        out = sim(**kwargs) if kwargs else sim()
    except TypeError:
        out = sim()
    # normaliza
    if len(out) >= 2:
        t, X_raw = out[0], out[1]
    else:
        raise RuntimeError("simulate() deve retornar pelo menos (t, X).")
    X = _as_7xN(X_raw)
    phase, incs = _phase_deg_and_incl_from_states(X)
    return np.asarray(t, float), X, phase, incs

# --------- seleção da ÚLTIMA ÓRBITA (preferência: por wraps de fase; fallback: por período) ----------
def _last_orbit_indices(t, phase_deg, X, elems=None):
    """
    Retorna slice de índices [i0:i1] da última órbita COMPLETA.
    Estratégia:
      1) Achar quebras (wraps) de fase: diff < -180 deg.
         Se houver >=2 wraps: usa o intervalo entre os DOIS ÚLTIMOS wraps (última órbita completa).
      2) Caso contrário: estima T = 2π sqrt(a^3/μ) usando 'a' terminal e recorta [t_end - T, t_end].
    """
    phase_deg = np.asarray(phase_deg, float)
    wraps = np.where(np.diff(phase_deg) < -180.0)[0]
    if wraps.size >= 2:
        i0 = wraps[-2] + 1
        i1 = wraps[-1] + 1  # slice exclusivo do último wrap (uma órbita completa)
        return slice(i0, i1)

    # Fallback por período
    if elems is None:
        elems = _elements_from_states(X)
    a_final = float(elems.a[-1])
    if a_final <= 0.0 or not np.isfinite(a_final):
        # fallback de emergência: pega 1 volta "aprox" pela média de velocidade e raio
        # (para LEO ~ 5400-6000 s). Usamos 6000 s como guardião.
        T_est = 6000.0
    else:
        n = np.sqrt(MU / (a_final**3))      # rad/s
        T_est = 2.0 * np.pi / n             # s

    t_end = float(t[-1])
    t_start = t_end - T_est
    if t_start <= t[0]:
        return slice(0, len(t))  # não cabe uma órbita completa; devolve tudo
    i0 = int(np.searchsorted(t, t_start, side="left"))
    i1 = len(t)
    return slice(i0, i1)

def _rephase_0_360(phi_slice):
    """Re-referencia a fase para iniciar em 0 e manter 0–360° na última órbita."""
    phi0 = float(phi_slice[0])
    x = (phi_slice - phi0) % 360.0
    return x

# ===================================== MAIN =====================================
if __name__ == "__main__":
    # Cores
    COLOR_UP   = "#1f77b4"  # azul
    COLOR_V    = "#2ca02c"  # verde
    COLOR_DOWN = "#d62728"  # vermelho

    # 1) Simulações (sem perturbações)
    t_up,   X_up,   phase_up,   incs_up   = _simulate_no_perturbations(S_UP)
    t_v,    X_v,    phase_v,    incs_v    = _simulate_no_perturbations(S_V)
    t_down, X_down, phase_down, incs_down = _simulate_no_perturbations(S_DOWN)

    # 2) Elementos (para Ω e também para fallback do período)
    E_up   = _elements_from_states(X_up)
    E_v    = _elements_from_states(X_v)
    E_down = _elements_from_states(X_down)

    # 3) Selecionar a ÚLTIMA ÓRBITA de cada satélite
    sl_up   = _last_orbit_indices(t_up,   phase_up,   X_up,   E_up)
    sl_v    = _last_orbit_indices(t_v,    phase_v,    X_v,    E_v)
    sl_down = _last_orbit_indices(t_down, phase_down, X_down, E_down)

    # Slices aplicados
    t_up_o,   X_up_o   = t_up[sl_up],     X_up[:, sl_up]
    t_v_o,    X_v_o    = t_v[sl_v],       X_v[:, sl_v]
    t_down_o, X_down_o = t_down[sl_down], X_down[:, sl_down]

    phase_up_o,   incs_up_o   = phase_up[sl_up],   incs_up[sl_up]
    phase_v_o,    incs_v_o    = phase_v[sl_v],     incs_v[sl_v]
    phase_down_o, incs_down_o = phase_down[sl_down], incs_down[sl_down]

    # Re-referenciar fase (0–360) para visual mais limpo na última órbita
    phase_up_o   = _rephase_0_360(phase_up_o)
    phase_v_o    = _rephase_0_360(phase_v_o)
    phase_down_o = _rephase_0_360(phase_down_o)

    # Recalcular elementos nas janelas (apenas para RAAN/a/e/i plotados)
    E_up_o   = _elements_from_states(X_up_o)
    E_v_o    = _elements_from_states(X_v_o)
    E_down_o = _elements_from_states(X_down_o)

    # --------------------------- 3D conjunto (última órbita) ---------------------------
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection="3d")
    _plot_earth_sphere(ax3d)
    ax3d.plot3D(X_up_o[0, :],   X_up_o[1, :],   X_up_o[2, :],   '-', color=COLOR_UP,   label="VH UP")
    ax3d.plot3D(X_v_o[0, :],    X_v_o[1, :],    X_v_o[2, :],    '-', color=COLOR_V,    label="V ONLY")
    ax3d.plot3D(X_down_o[0, :], X_down_o[1, :], X_down_o[2, :], '-', color=COLOR_DOWN, label="VH DOWN")
    _force_equal_3d_limits(ax3d, [X_up_o, X_v_o, X_down_o])
    ax3d.set_xlabel("x [km]"); ax3d.set_ylabel("y [km]"); ax3d.set_zlabel("z [km]")
    ax3d.set_title("Órbitas — Três satélites (apenas a ÚLTIMA órbita)")
    ax3d.legend(); plt.show()

    # --------------------------- Massa × Tempo (última órbita) ---------------------------
    plt.figure()
    if X_up_o.shape[0]   >= 7: plt.plot(t_up_o,   X_up_o[6, :],   '-', color=COLOR_UP,   label="VH UP")
    if X_v_o.shape[0]    >= 7: plt.plot(t_v_o,    X_v_o[6, :],    '-', color=COLOR_V,    label="V ONLY")
    if X_down_o.shape[0] >= 7: plt.plot(t_down_o, X_down_o[6, :], '-', color=COLOR_DOWN, label="VH DOWN")
    plt.xlabel("Tempo [s]"); plt.ylabel("Massa [kg]")
    plt.title("Consumo de Propelente — ÚLTIMA órbita")
    plt.grid(alpha=0.3); plt.legend(); plt.show()

    # --------------------------- Inclinação × Fase (última órbita) ---------------------------
    def _plot_i_vs_phase(phi, inc, color, label, ax):
        dn = np.diff(phi); wraps = np.where(dn < -180.0)[0]
        start = 0; first = True
        for w in wraps:
            ax.plot(phi[start:w+1], inc[start:w+1], color=color, label=(label if first else None))
            first = False; start = w + 1
        ax.plot(phi[start:], inc[start:], color=color, label=(label if first else None))

    fig2, ax2 = plt.subplots()
    _plot_i_vs_phase(phase_up_o,   incs_up_o,   COLOR_UP,   "VH UP",   ax2)
    _plot_i_vs_phase(phase_v_o,    incs_v_o,    COLOR_V,    "V ONLY",  ax2)
    _plot_i_vs_phase(phase_down_o, incs_down_o, COLOR_DOWN, "VH DOWN", ax2)
    ax2.set_xlim(0, 360); ax2.grid(True)
    ax2.set_xlabel(r'Fase (ν ou $u$) [deg]'); ax2.set_ylabel(r'$i$ [deg]')
    ax2.set_title("Inclinação × Fase — ÚLTIMA órbita")
    ax2.legend(); plt.show()

    # --------------------------- RAAN (Ω) desenrolado × dias (última órbita) ---------------------------
    plt.figure()
    plt.plot(t_up_o/86400.0,   _unwrap_deg(E_up_o.Omega_deg),   '-', color=COLOR_UP,   label="VH UP")
    plt.plot(t_v_o/86400.0,    _unwrap_deg(E_v_o.Omega_deg),    '-', color=COLOR_V,    label="V ONLY")
    plt.plot(t_down_o/86400.0, _unwrap_deg(E_down_o.Omega_deg), '-', color=COLOR_DOWN, label="VH DOWN")
    plt.xlabel("Tempo [dias]"); plt.ylabel("Ω (graus)")
    plt.title("RAAN (Ω) desenrolado — ÚLTIMA órbita")
    plt.grid(True); plt.legend(); plt.show()

    # --------------------------- Painel a, e, i, Ω (última órbita) ---------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    # a (km)
    axs[0,0].plot(t_up_o,   E_up_o.a,   color=COLOR_UP,   label="VH UP")
    axs[0,0].plot(t_v_o,    E_v_o.a,    color=COLOR_V,    label="V ONLY")
    axs[0,0].plot(t_down_o, E_down_o.a, color=COLOR_DOWN, label="VH DOWN")
    axs[0,0].set_title("a (km)"); axs[0,0].set_xlabel("t (s)"); axs[0,0].grid(True)

    # e (-)
    axs[0,1].plot(t_up_o,   E_up_o.e,   color=COLOR_UP)
    axs[0,1].plot(t_v_o,    E_v_o.e,    color=COLOR_V)
    axs[0,1].plot(t_down_o, E_down_o.e, color=COLOR_DOWN)
    axs[0,1].set_title("e"); axs[0,1].set_xlabel("t (s)"); axs[0,1].grid(True)

    # i (deg)
    axs[1,0].plot(t_up_o,   E_up_o.i_deg,   color=COLOR_UP)
    axs[1,0].plot(t_v_o,    E_v_o.i_deg,    color=COLOR_V)
    axs[1,0].plot(t_down_o, E_down_o.i_deg, color=COLOR_DOWN)
    axs[1,0].set_title("i (graus)"); axs[1,0].set_xlabel("t (s)"); axs[1,0].grid(True)

    # Ω (deg) desenrolado
    axs[1,1].plot(t_up_o,   _unwrap_deg(E_up_o.Omega_deg),   color=COLOR_UP)
    axs[1,1].plot(t_v_o,    _unwrap_deg(E_v_o.Omega_deg),    color=COLOR_V)
    axs[1,1].plot(t_down_o, _unwrap_deg(E_down_o.Omega_deg), color=COLOR_DOWN)
    axs[1,1].set_title("Ω (graus, desenrolado)"); axs[1,1].set_xlabel("t (s)"); axs[1,1].grid(True)

    # Legenda única
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False)
    fig.suptitle("Comparativo de elementos — ÚLTIMA órbita", fontsize=12)
    plt.show()
