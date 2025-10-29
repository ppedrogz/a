# main_constelacao_joint.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import inspect

# Satélites (cada módulo precisa expor simulate())
import sat_vh_down as S_DOWN
import sat_vh_up   as S_UP
import sat_v_only  as S_V

# Utils do projeto
from utils.visualization import ElementsSeries, plot_classic_orbital_elements  # (usado p/ estilos)
from utils.orbital_elements import *
from utils.orbitalElementsOperations import *

# ------------------------------- Constantes -------------------------------
EARTH_RADIUS_KM = 6378.0
MU = 3.986e5  # km^3/s^2

# ------------------------------- Helpers ---------------------------------
def _as_7xN(X):
    """Garante 7xN (x,y,z,vx,vy,vz,m). Aceita 7xN, Nx7, 6xN, Nx6."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X deve ser 2D.")
    if X.shape[0] in (6, 7) and X.shape[1] not in (6, 7):
        X7 = X
    elif X.shape[1] in (6, 7) and X.shape[0] not in (6, 7):
        X7 = X.T
    else:
        X7 = X
    if X7.shape[0] == 6:
        X7 = np.vstack([X7, np.full((1, X7.shape[1]), np.nan)])
    return X7

def _unwrap_deg(a_deg):
    return np.degrees(np.unwrap(np.radians(np.asarray(a_deg, float))))

def _phase_deg_from_states(X):
    """Fase robusta (ν ou u) e inclinação em graus a partir de X."""
    N = X.shape[1]
    phase = np.empty(N); incs = np.empty(N)
    for k in range(N):
        r = X[0:3, k]; v = X[3:6, k]
        e_now = get_eccentricity(r, v, MU)
        phase[k] = get_argument_of_latitude(r, v, MU) if e_now < 1e-5 else get_true_anomaly(r, v, MU)
        incs[k]  = get_inclination(r, v, MU)
    return phase, incs

def _elements_from_states(X):
    """Calcula ElementsSeries (a,e,i,Ω,ω,ν,u,l_true,energia) a partir de X."""
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
    """Zera flags comuns de perturbações, se existirem."""
    for attr in ("_USE_J2", "_USE_J22", "_DRAG_ON", "_J2_ON", "_J22_ON", "J2_ON", "J22_ON", "DRAG_ON"):
        if hasattr(mod, attr):
            try: setattr(mod, attr, False)
            except Exception: pass

def _simulate_no_perturbations(mod):
    """Chama mod.simulate() sem ativar perturbações; tolera assinaturas diferentes."""
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

    if len(out) >= 4:
        t, X_raw, phase_deg, incs_deg = out[0], out[1], out[2], out[3]
    else:
        t, X_raw = out[0], out[1]
        Xtmp = _as_7xN(X_raw)
        phase_deg, incs_deg = _phase_deg_from_states(Xtmp)
    X = _as_7xN(X_raw)
    return np.asarray(t, float), X, np.asarray(phase_deg, float), np.asarray(incs_deg, float)

# ------------------------------- Execução ---------------------------------
if __name__ == "__main__":
    # Cores
    COLOR_UP   = "#1f77b4"  # azul
    COLOR_V    = "#2ca02c"  # verde
    COLOR_DOWN = "#d62728"  # vermelho

    # 1) Simular os três (sem perturbações)
    t_up,   X_up,   phase_up,   incs_up   = _simulate_no_perturbations(S_UP)
    t_v,    X_v,    phase_v,    incs_v    = _simulate_no_perturbations(S_V)
    t_down, X_down, phase_down, incs_down = _simulate_no_perturbations(S_DOWN)

    # 2) Elements para cada um (para Ω, a, e, i, etc.)
    E_up   = _elements_from_states(X_up)
    E_v    = _elements_from_states(X_v)
    E_down = _elements_from_states(X_down)

    # --------------------------- GRÁFICO 3D (CONJUNTO) ---------------------------
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection="3d")
    _plot_earth_sphere(ax3d)
    ax3d.plot3D(X_up[0, :],   X_up[1, :],   X_up[2, :],   '-', color=COLOR_UP,   label="VH UP")
    ax3d.plot3D(X_v[0, :],    X_v[1, :],    X_v[2, :],    '-', color=COLOR_V,    label="V ONLY")
    ax3d.plot3D(X_down[0, :], X_down[1, :], X_down[2, :], '-', color=COLOR_DOWN, label="VH DOWN")
    _force_equal_3d_limits(ax3d, [X_up, X_v, X_down])
    ax3d.set_xlabel("x [km]"); ax3d.set_ylabel("y [km]"); ax3d.set_zlabel("z [km]")
    ax3d.set_title("Órbitas — Três satélites (sem perturbações)")
    ax3d.legend(); plt.show()

    # --------------------------- MASSA × TEMPO (CONJUNTO) ---------------------------
    plt.figure()
    if X_up.shape[0]   >= 7: plt.plot(t_up,   X_up[6, :],   '-', color=COLOR_UP,   label="VH UP")
    if X_v.shape[0]    >= 7: plt.plot(t_v,    X_v[6, :],    '-', color=COLOR_V,    label="V ONLY")
    if X_down.shape[0] >= 7: plt.plot(t_down, X_down[6, :], '-', color=COLOR_DOWN, label="VH DOWN")
    plt.xlabel("Tempo [s]"); plt.ylabel("Massa [kg]")
    plt.title("Consumo de Propelente — Três satélites (sem perturbações)")
    plt.grid(alpha=0.3); plt.legend(); plt.show()

    # --------------------------- i × FASE (CONJUNTO, segmentado) ---------------------------
    def _plot_i_vs_phase(phi, inc, color, label, ax):
        dn = np.diff(phi); wraps = np.where(dn < -180.0)[0]
        start = 0; first = True
        for w in wraps:
            ax.plot(phi[start:w+1], inc[start:w+1], color=color, label=(label if first else None))
            first = False; start = w + 1
        ax.plot(phi[start:], inc[start:], color=color, label=(label if first else None))

    fig2, ax2 = plt.subplots()
    _plot_i_vs_phase(phase_up,   incs_up,   COLOR_UP,   "VH UP",   ax2)
    _plot_i_vs_phase(phase_v,    incs_v,    COLOR_V,    "V ONLY",  ax2)
    _plot_i_vs_phase(phase_down, incs_down, COLOR_DOWN, "VH DOWN", ax2)
    ax2.set_xlim(0, 360); ax2.grid(True)
    ax2.set_xlabel(r'Fase (ν ou $u$) [deg]'); ax2.set_ylabel(r'$i$ [deg]')
    ax2.set_title("Inclinação × Fase — Três satélites (sem perturbações)")
    ax2.legend(); plt.show()

    # --------------------------- Ω desenrolado × DIAS (CONJUNTO) ---------------------------
    t_up_d   = t_up/86400.0
    t_v_d    = t_v/86400.0
    t_down_d = t_down/86400.0
    plt.figure()
    plt.plot(t_up_d,   _unwrap_deg(E_up.Omega_deg),   '-', color=COLOR_UP,   label="VH UP")
    plt.plot(t_v_d,    _unwrap_deg(E_v.Omega_deg),    '-', color=COLOR_V,    label="V ONLY")
    plt.plot(t_down_d, _unwrap_deg(E_down.Omega_deg), '-', color=COLOR_DOWN, label="VH DOWN")
    plt.xlabel("Tempo [dias]"); plt.ylabel("Ω (graus)")
    plt.title("RAAN (Ω) desenrolado — Três satélites (sem perturbações)")
    plt.grid(True); plt.legend(); plt.show()

    # --------------------------- PAINEL: a, e, i, Ω (CONJUNTO) ---------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    # a (km)
    axs[0,0].plot(t_up,   E_up.a,   color=COLOR_UP,   label="VH UP")
    axs[0,0].plot(t_v,    E_v.a,    color=COLOR_V,    label="V ONLY")
    axs[0,0].plot(t_down, E_down.a, color=COLOR_DOWN, label="VH DOWN")
    axs[0,0].set_title("a (km)"); axs[0,0].set_xlabel("t (s)"); axs[0,0].grid(True)

    # e (-)
    axs[0,1].plot(t_up,   E_up.e,   color=COLOR_UP)
    axs[0,1].plot(t_v,    E_v.e,    color=COLOR_V)
    axs[0,1].plot(t_down, E_down.e, color=COLOR_DOWN)
    axs[0,1].set_title("e"); axs[0,1].set_xlabel("t (s)"); axs[0,1].grid(True)

    # i (deg)
    axs[1,0].plot(t_up,   E_up.i_deg,   color=COLOR_UP)
    axs[1,0].plot(t_v,    E_v.i_deg,    color=COLOR_V)
    axs[1,0].plot(t_down, E_down.i_deg, color=COLOR_DOWN)
    axs[1,0].set_title("i (graus)"); axs[1,0].set_xlabel("t (s)"); axs[1,0].grid(True)

    # Ω (deg) desenrolado
    axs[1,1].plot(t_up,   _unwrap_deg(E_up.Omega_deg),   color=COLOR_UP)
    axs[1,1].plot(t_v,    _unwrap_deg(E_v.Omega_deg),    color=COLOR_V)
    axs[1,1].plot(t_down, _unwrap_deg(E_down.Omega_deg), color=COLOR_DOWN)
    axs[1,1].set_title("Ω (graus, desenrolado)"); axs[1,1].set_xlabel("t (s)"); axs[1,1].grid(True)

    # Legenda única
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False)
    fig.suptitle("Comparativo de elementos — Três satélites (sem perturbações)", fontsize=12)
    plt.show()
