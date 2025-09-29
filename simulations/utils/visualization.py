import matplotlib.pyplot as plt
from utils.types import OrbitalElements
import numpy as np
from cycler import cycler

# --------- helpers p/ plot ----------

import numpy as np
import matplotlib.pyplot as plt
from utils.types import OrbitalElements

# -------------------- constantes p/ suavização e máscaras --------------------
_WIN_ANG_SMOOTH_F = 250            # janela ~ N/250 (mín. 5)
_EPS_I_PLOT_DEG   = 0.05           # ~equatorial para ZOH (graus)
_EPS_E_PLOT       = 2e-3           # ~circular para ZOH de ω

# ------------------------------ helpers genéricos -----------------------------
def _series_from(orbital_elementss: list[OrbitalElements]):
    a  = [el.major_axis for el in orbital_elementss]
    e  = [el.eccentricity for el in orbital_elementss]
    i  = [el.inclination for el in orbital_elementss]         # deg
    Om = [el.ascending_node for el in orbital_elementss]      # deg
    w  = [el.argument_of_perigee for el in orbital_elementss] # deg
    nu = [el.true_anomaly for el in orbital_elementss]        # deg (ou u/λ conforme caso)
    return np.array(a,float), np.array(e,float), np.array(i,float), \
           np.array(Om,float), np.array(w,float), np.array(nu,float)

def _process_e_for_plot(e: np.ndarray):
    # Pode aplicar filtro leve, clip etc. Aqui deixo simples/estável.
    e = np.asarray(e, float)
    e = np.clip(e, 0.0, None)
    return e

def _contiguous_from_prev(x_deg):
    """unwrap simples em graus preservando continuidade local."""
    x = np.asarray(x_deg, dtype=float)
    if x.size == 0: return x
    out = x.copy()
    for k in range(1, len(x)):
        d = out[k] - out[k-1]
        if d >  180.0: out[k] -= 360.0
        if d < -180.0: out[k] += 360.0
    return out

def _rolling_mean(x, win):
    if win <= 1: return x
    win = int(win)
    kernel = np.ones(win)/win
    return np.convolve(x, kernel, mode='same')

def _to_0_360_no_edge_jump(x_cont):
    """Converte série contínua (unwrap) para [0,360) sem criar degrau na borda."""
    return np.mod(np.asarray(x_cont, float), 360.0)

def _zoh_when_masked(y, mask):
    """
    Zero-Order Hold quando mask[k] == True: mantém último valor válido.
    Útil para trechos onde a grandeza é indefinida (equatorial/circular).
    """
    y = np.asarray(y, float).copy()
    mask = np.asarray(mask, bool)
    if y.size == 0: return y
    last = y[0]
    for k in range(len(y)):
        if mask[k]:
            y[k] = last
        else:
            last = y[k]
    return y

# ---------------------- pipeline de ângulos p/ visualização -------------------
def _process_angles_for_plot(inc_deg, Om_deg, w_deg, nu_deg,
                             *,
                             show_mod360: bool = False,
                             use_zoh_on_equatorial: bool = True,
                             use_zoh_on_circular: bool = True,
                             e_series: np.ndarray | None = None,
                             e_eps: float = _EPS_E_PLOT):
    """
    - unwrap + média móvel para Ω, ω, ν
    - ZOH em casos equatoriais/circulares (baseado em i e e)
    - opcionalmente, volta para [0,360) no fim
    """
    inc_deg = np.asarray(inc_deg, float)
    Om = _contiguous_from_prev(Om_deg)
    w  = _contiguous_from_prev(w_deg)
    nu = _contiguous_from_prev(nu_deg)

    N = len(inc_deg)
    win = max(5, (N // _WIN_ANG_SMOOTH_F) | 1)

    Om = _rolling_mean(Om, win)
    w  = _rolling_mean(w,  win)
    nu = _rolling_mean(nu, win)

    # máscaras
    dist_eq = np.minimum(np.abs(inc_deg), np.abs(180.0 - inc_deg))
    mask_eq = (dist_eq < _EPS_I_PLOT_DEG) if use_zoh_on_equatorial else np.zeros_like(inc_deg, bool)
    if e_series is None or (not use_zoh_on_circular):
        mask_circ = np.zeros_like(inc_deg, bool)
    else:
        mask_circ = (np.asarray(e_series) <= e_eps)

    # ZOH: quando equatorial -> Ω ZOH; quando circular -> ω ZOH
    Om_plot = _zoh_when_masked(Om, mask_eq) if use_zoh_on_equatorial else Om
    w_plot  = _zoh_when_masked(w,  mask_circ) if use_zoh_on_circular else w
    nu_plot = nu  # já contínua; em circular ela representa u via elementos

    if show_mod360:
        Om_plot = np.mod(Om_plot, 360.0)
        w_plot  = np.mod(w_plot,  360.0)
        nu_plot = np.mod(nu_plot, 360.0)

    return Om_plot, w_plot, nu_plot

# ----------------------------- função principal ------------------------------
def plot_classic_orbital_elements(t: np.ndarray,
                                  orbital_elementss: list[OrbitalElements],
                                  *,
                                  show_mod360: bool = False,
                                  e_series: np.ndarray | None = None,
                                  e_eps: float = _EPS_E_PLOT,
                                  use_zoh_on_equatorial: bool = True,
                                  use_zoh_on_circular: bool = True):
    """
    Plota (a, e, i, Ω, ω, ν) com ângulos processados (unwrap, suavização e ZOH)
    e inclui também o gráfico do argumento de latitude u = ν + ω (robusto).
    """
    # séries brutas
    a, e, inc, Om, w, nu = _series_from(orbital_elementss)
    e_plot = _process_e_for_plot(e)

    # ângulos processados (Ω usa série processada!)
    Om_plot, w_plot, nu_plot = _process_angles_for_plot(
        inc, Om, w, nu,
        show_mod360=show_mod360,
        use_zoh_on_equatorial=use_zoh_on_equatorial,
        use_zoh_on_circular=use_zoh_on_circular,
        e_series=(e if e_series is None else np.asarray(e_series, float)),
        e_eps=e_eps
    )

    # ---------- Argumento de latitude: u = ν + ω ----------
    u_raw = (w + nu) % 360.0
    N = len(u_raw)
    win_ang = max(5, (N // _WIN_ANG_SMOOTH_F) | 1)

    u_cont = _contiguous_from_prev(u_raw)
    u_cont = _rolling_mean(u_cont, win_ang)
    u_mod  = _to_0_360_no_edge_jump(u_cont)

    # máscara ~equatorial (u indefinido) → ZOH
    dist_eq = np.minimum(np.abs(inc), np.abs(180.0 - inc))
    mask_eq = (dist_eq < _EPS_I_PLOT_DEG)
    u_plot = _zoh_when_masked(u_mod, mask_eq)

    # ------------------ plots ------------------
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # a
    axs[0, 0].plot(t, a, label='Major Axis')
    axs[0, 0].set_title('Major Axis')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Major Axis (km)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # e
    axs[0, 1].plot(t, e_plot, label='Eccentricity', color='orange')
    axs[0, 1].set_title('Eccentricity')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Eccentricity')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # i
    axs[1, 0].plot(t, inc, label='Inclination', color='green')
    axs[1, 0].set_title('Inclination')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Inclination (degrees)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Ω — usa série processada Om_plot (sem saltos 360→0)
    axs[1, 1].plot(t, Om_plot, label='Ascending Node (RAAN)')
    axs[1, 1].set_title('Ascending Node (RAAN)')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Ascending Node (degrees)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # u
    axs[2, 0].plot(t, u_plot, label='Argument of Latitude (u)', color='purple')
    axs[2, 0].set_title('Argument of Latitude (u)')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylabel('Argument of Latitude (deg)')
    axs[2, 0].set_ylim(0.0, 360.0)
    axs[2, 0].set_yticks([0, 60, 120, 180, 240, 300, 360])
    axs[2, 0].grid(True)
    axs[2, 0].legend()

    # ν — já contínua pelo pipeline (_process_angles_for_plot)
    axs[2, 1].plot(t, nu_plot, label='True Anomaly', color='brown')
    axs[2, 1].set_title('True Anomaly')
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].set_ylabel('True Anomaly (degrees)')
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()



def plot_3D_view(
        X,
        plot_earth: bool = True,
        earth_radius: float = 6378.0,
        earth_color: str = 'blue',
        earth_alpha: float = 0.3
        ):
    plt.figure()
    if plot_earth:
        ax = plt.axes(projection='3d')
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = earth_radius * np.cos(u)*np.sin(v)
        y = earth_radius * np.sin(u)*np.sin(v)
        z = earth_radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color=earth_color, alpha=earth_alpha)

    ax.plot3D(X[0, :], X[1, :], X[2, :], 'b-')
    ax.set_title('Orbit Propagation')
    ax.axis('equal')
    plt.show()

def plot_3D_overlay(
        *Xs,
        plot_earth: bool = True,
        earth_radius: float = 6378.0,
        earth_color: str = 'blue',
        earth_alpha: float = 0.3,
        orbit_marker: str = '-'
        ):
    plt.figure()
    plt.style.use('bmh')
    if plot_earth:
        ax = plt.axes(projection='3d')
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = earth_radius * np.cos(u)*np.sin(v)
        y = earth_radius * np.sin(u)*np.sin(v)
        z = earth_radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color=earth_color, alpha=earth_alpha)

    markers = ['-', '--', ':', '-.']
    for i in range(len(Xs)):
        X = Xs[i]
        ax.plot3D(X[0, :], X[1, :], X[2, :], markers[i], linewidth=3)
    ax.set_title('Orbit Propagation')
    ax.axis('equal')
    plt.show()
