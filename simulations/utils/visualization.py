import matplotlib.pyplot as plt
from utils.types import OrbitalElements
import numpy as np
from cycler import cycler

# --------- helpers p/ plot ----------

def _unwrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    """desembrulha em rad e volta p/ deg (contínuo), útil p/ Ω, ω, ν"""
    ang = np.asarray(angle_deg, dtype=float)
    return np.degrees(np.unwrap(np.radians(ang)))

def _rolling_mean(y: np.ndarray, win: int) -> np.ndarray:
    """média móvel simples (janela ímpar)"""
    win = int(max(1, win | 1))  # força ímpar
    if win <= 1:
        return np.asarray(y, dtype=float)
    k = np.ones(win, dtype=float) / win
    return np.convolve(np.asarray(y, float), k, mode="same")

def _mod360_cont(angle_deg_cont: np.ndarray) -> np.ndarray:
    """aplica módulo 360 preservando continuidade local"""
    return np.mod(angle_deg_cont, 360.0)

def _mask_if(cond: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.array(y, dtype=float, copy=True)
    z[cond] = np.nan
    return z

# ===== NOVOS helpers para continuidade/ZOH/exibição =====

def _contiguous_from_prev(raw_deg: np.ndarray) -> np.ndarray:
    """
    Constrói série contínua a partir de valores em [0,360):
    faz o wrap pelo vizinho mais próximo (±180°).
    """
    raw_deg = np.asarray(raw_deg, float)
    if raw_deg.size == 0:
        return raw_deg.copy()
    cont = np.empty_like(raw_deg)
    cont[0] = raw_deg[0]
    for k in range(1, raw_deg.size):
        delta = ((raw_deg[k] - raw_deg[k-1] + 180.0) % 360.0) - 180.0
        cont[k] = cont[k-1] + delta
    return cont

def _to_0_360_no_edge_jump(y_cont: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Converte série contínua para [0,360) na exibição, “colando” 360 com 0.
    """
    y_mod = np.mod(y_cont, 360.0)
    y_mod[np.abs(360.0 - y_mod) < eps] = 0.0
    return y_mod

def _zoh_when_masked(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Hold-last (ZOH) quando mask=True. Útil para Ω e ω quando indefinidos
    (i≈0°/180° para Ω; e≈0 para ω).
    """
    y = np.array(y, float, copy=True)
    if y.size == 0:
        return y
    valid = ~mask & ~np.isnan(y)
    last = y[valid][0] if np.any(valid) else 0.0
    for k in range(y.size):
        if mask[k] or np.isnan(y[k]):
            y[k] = last
        else:
            last = y[k]
    return y

# ---------------- parâmetros “de plot” (não afetam dinâmica) ---------------
# pode ajustar conforme o ruído da tua integração
_EPS_I_PLOT_DEG   = 1e-6      # máscara equatorial só quando i é extremamente pequeno
_EPS_E_PLOT       = 0.0       # NÃO "snap-to-zero": mostra e pequeno
_WIN_E_SMOOTH_F   = 200       # janela ~N/_WIN_E_SMOOTH_F para e
_WIN_ANG_SMOOTH_F = 1000      # janela ~N/_WIN_ANG_SMOOTH_F para ângulos (menos suavização)

def _series_from(orbital_elementss: list[OrbitalElements]):
    a  = np.array([e.major_axis           for e in orbital_elementss], float)
    ec = np.array([e.eccentricity         for e in orbital_elementss], float)
    inc= np.array([e.inclination          for e in orbital_elementss], float)     # deg
    Om = np.array([e.ascending_node       for e in orbital_elementss], float)     # deg
    w  = np.array([e.argument_of_perigee  for e in orbital_elementss], float)     # deg
    nu = np.array([e.true_anomaly         for e in orbital_elementss], float)     # deg
    return a, ec, inc, Om, w, nu

def _process_angles_for_plot(inc_deg, Om_deg, w_deg, nu_deg,
                             *, show_mod360: bool = False,
                             use_zoh_on_equatorial: bool = True,
                             use_zoh_on_circular: bool = True,
                             e_series: np.ndarray | None = None,
                             e_eps: float = _EPS_E_PLOT):
    """
    - Ω contínuo (sem %360 no traço), com ZOH quando i≈0°/180° (indefinido).
    - ω contínuo, com ZOH quando e≈0 (indefinido em circular) e quando i≈0°/180° (equatorial).
    - ν contínuo (sem %360).
    - show_mod360=True remapeia no FINAL para [0,360) sem criar degrau.
    """
    N = len(inc_deg)
    win_ang = max(5, (N // _WIN_ANG_SMOOTH_F) | 1)

    # continuidade sem %360 (evita degrau)
    Om_cont = _contiguous_from_prev(Om_deg)
    w_cont  = _contiguous_from_prev(w_deg)
    nu_cont = _contiguous_from_prev(nu_deg)

    # suavização leve
    Om_cont = _rolling_mean(Om_cont, win_ang)
    w_cont  = _rolling_mean(w_cont,  win_ang)
    nu_cont = _rolling_mean(nu_cont, win_ang)

    # máscaras: equatorial (Ω e ω indefinidos) e circular (ω indefinido)
    dist_eq  = np.minimum(np.abs(inc_deg), np.abs(180.0 - inc_deg))
    mask_eq  = (dist_eq < _EPS_I_PLOT_DEG)

    if e_series is None:
        mask_circ = np.zeros_like(inc_deg, dtype=bool)
    else:
        e_s = np.asarray(e_series, float)
        mask_circ = (e_s < e_eps)

    # ZOH quando indefinidos
    if use_zoh_on_equatorial:
        Om_cont = _zoh_when_masked(Om_cont, mask_eq)  # Ω indef. em equatorial
        w_cont  = _zoh_when_masked(w_cont,  mask_eq)  # ω idem
    else:
        Om_cont = _mask_if(mask_eq, Om_cont)
        w_cont  = _mask_if(mask_eq, w_cont)

    if use_zoh_on_circular:
        w_cont = _zoh_when_masked(w_cont, mask_circ)  # ω indef. em circular
    else:
        w_cont = _mask_if(mask_circ, w_cont)

    # saída: contínuo ou [0,360) sem degrau
    if show_mod360:
        Om_plot = _to_0_360_no_edge_jump(Om_cont)
        w_plot  = _to_0_360_no_edge_jump(w_cont)
        nu_plot = _to_0_360_no_edge_jump(nu_cont)
    else:
        Om_plot, w_plot, nu_plot = Om_cont, w_cont, nu_cont

    return Om_plot, w_plot, nu_plot

def _process_e_for_plot(e):
    """
    Suaviza e (sem forçar a zero por padrão).
    """
    e = np.asarray(e, float)
    N = len(e)
    win_e = max(5, (N // _WIN_E_SMOOTH_F) | 1)
    e_smooth = _rolling_mean(e, win_e)
    # sem snap-to-zero; se quiser, use: np.where(e_smooth < 2e-3, 0.0, e_smooth)
    return e_smooth

def plot_classic_orbital_elements(t: np.typing.NDArray, orbital_elementss: list[OrbitalElements]):
    """
    Plota elementos orbitais clássicos com tratamentos de plot para:
    - Ω: série contínua (sem %360), ZOH quando i≈0°/180°
    - e: suavização (sem snap-to-zero por padrão)
    - ω, ν: séries contínuas (sem %360) para eliminar degraus 360→0
    """
    a, e, inc, Om, w, nu = _series_from(orbital_elementss)
    e_plot = _process_e_for_plot(e)  # processa e primeiro (fornecer a série para ω)
    Om_plot, w_plot, nu_plot = _process_angles_for_plot(
        inc, Om, w, nu,
        show_mod360=False,              # contínuo: melhor p/ ver precessão de Ω
        use_zoh_on_equatorial=True,     # segura Ω/ω quando i≈0°/180°
        use_zoh_on_circular=True,       # segura ω quando e≈0
        e_series=e,                     # série crua de e para detecção de circular
        e_eps=_EPS_E_PLOT
    )

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    axs[0, 0].plot(t, a, label='Major Axis')
    axs[0, 0].set_title('Major Axis')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Major Axis (km)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(t, e_plot, label='Eccentricity', color='orange')
    axs[0, 1].set_title('Eccentricity')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Eccentricity')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(t, inc, label='Inclination', color='green')
    axs[1, 0].set_title('Inclination')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Inclination (degrees)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(t, Om_plot, label='Ascending Node', color='red')
    axs[1, 1].set_title('Ascending Node')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Ascending Node (degrees)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    axs[2, 0].plot(t, w_plot, label='Argument of Perigee', color='purple')
    axs[2, 0].set_title('Argument of Perigee')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylabel('Argument of Perigee (degrees)')
    axs[2, 0].grid(True)
    axs[2, 0].legend()

    axs[2, 1].plot(t, nu_plot, label='True Anomaly', color='brown')
    axs[2, 1].set_title('True Anomaly')
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].set_ylabel('True Anomaly (degrees)')
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()


def plot_classic_orbital_elements_overlay(*orbital_elementss_lists: list[np.typing.NDArray, list[OrbitalElements]]):
    """
    Recebe tuplas (t, orbital_elementss) e sobrepõe os elementos com os mesmos tratamentos.
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    for idx, orbital_elementss_list in enumerate(orbital_elementss_lists):
        t = orbital_elementss_list[0]
        orbital_elementss = orbital_elementss_list[1]

        a, e, inc, Om, w, nu = _series_from(orbital_elementss)
        e_plot = _process_e_for_plot(e)
        Om_plot, w_plot, nu_plot = _process_angles_for_plot(
            inc, Om, w, nu,
            show_mod360=False,
            use_zoh_on_equatorial=True,
            use_zoh_on_circular=True,
            e_series=e,
            e_eps=_EPS_E_PLOT
        )

        axs[0, 0].plot(t, a, label=f'Major Axis #{idx+1}')
        axs[0, 1].plot(t, e_plot, label=f'Eccentricity #{idx+1}')
        axs[1, 0].plot(t, inc, label=f'Inclination #{idx+1}')
        axs[1, 1].plot(t, Om_plot, label=f'Ascending Node #{idx+1}')
        axs[2, 0].plot(t, w_plot, label=f'Argument of Perigee #{idx+1}')
        axs[2, 1].plot(t, nu_plot, label=f'True Anomaly #{idx+1}')

    axs[0, 0].set_title('Major Axis');      axs[0, 0].set_xlabel('Time (s)'); axs[0, 0].set_ylabel('Major Axis (km)'); axs[0, 0].grid(True); axs[0, 0].legend()
    axs[0, 1].set_title('Eccentricity');    axs[0, 1].set_xlabel('Time (s)'); axs[0, 1].set_ylabel('Eccentricity');     axs[0, 1].grid(True); axs[0, 1].legend()
    axs[1, 0].set_title('Inclination');     axs[1, 0].set_xlabel('Time (s)'); axs[1, 0].set_ylabel('Inclination (deg)');axs[1, 0].grid(True); axs[1, 0].legend()
    axs[1, 1].set_title('Ascending Node');  axs[1, 1].set_xlabel('Time (s)'); axs[1, 1].set_ylabel('Ascending Node (deg)'); axs[1, 1].grid(True); axs[1, 1].legend()
    axs[2, 0].set_title('Argument of Perigee'); axs[2, 0].set_xlabel('Time (s)'); axs[2, 0].set_ylabel('Argument of Perigee (deg)'); axs[2, 0].grid(True); axs[2, 0].legend()
    axs[2, 1].set_title('True Anomaly');    axs[2, 1].set_xlabel('Time (s)'); axs[2, 1].set_ylabel('True Anomaly (deg)'); axs[2, 1].grid(True); axs[2, 1].legend()

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
