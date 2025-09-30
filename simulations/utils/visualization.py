import matplotlib.pyplot as plt
from utils.types import OrbitalElements
import numpy as np
from cycler import cycler

# --------- helpers p/ plot ----------

def _unwrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    ang = np.asarray(angle_deg, dtype=float)
    return np.degrees(np.unwrap(np.radians(ang)))

def _rolling_mean(y: np.ndarray, win: int) -> np.ndarray:
    win = int(max(1, win | 1))  # força ímpar
    if win <= 1:
        return np.asarray(y, dtype=float)
    k = np.ones(win, dtype=float) / win
    return np.convolve(np.asarray(y, float), k, mode="same")

def _mod360_cont(angle_deg_cont: np.ndarray) -> np.ndarray:
    return np.mod(angle_deg_cont, 360.0)

def _mask_if(cond: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.array(y, dtype=float, copy=True)
    z[cond] = np.nan
    return z
# ===== NOVOS helpers para continuidade/ZOH/exibição =====

def _contiguous_from_prev(raw_deg: np.ndarray) -> np.ndarray:
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
    y_mod = np.mod(y_cont, 360.0)
    y_mod[np.abs(360.0 - y_mod) < eps] = 0.0
    return y_mod

def _zoh_when_masked(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
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

_EPS_I_PLOT_DEG   = 1e-6
_EPS_E_PLOT       = 2e-3    # mesmo do orbitalElementsOperations
_WIN_E_SMOOTH_F   = 200
_WIN_ANG_SMOOTH_F = 1000

def _series_from(orbital_elementss: list[OrbitalElements]):
    a  = np.array([e.major_axis           for e in orbital_elementss], float)
    ec = np.array([e.eccentricity         for e in orbital_elementss], float)
    inc= np.array([e.inclination          for e in orbital_elementss], float)
    Om = np.array([e.ascending_node       for e in orbital_elementss], float)
    w  = np.array([e.argument_of_perigee  for e in orbital_elementss], float)
    nu = np.array([e.true_anomaly         for e in orbital_elementss], float)
    return a, ec, inc, Om, w, nu

def _process_angles_for_plot(inc_deg, Om_deg, w_deg, nu_deg,
                             *, show_mod360: bool = False,
                             use_zoh_on_equatorial: bool = True,
                             use_zoh_on_circular: bool = True,
                             e_series: np.ndarray | None = None,
                             e_eps: float = _EPS_E_PLOT):
    N = len(inc_deg)
    win_ang = max(5, (N // _WIN_ANG_SMOOTH_F) | 1)

    # 1) desenrolar (contiguidade)
    Om_cont = _contiguous_from_prev(Om_deg)
    w_cont  = _contiguous_from_prev(w_deg)
    nu_cont = _contiguous_from_prev(nu_deg)

    # 3) máscaras de regimes degenerados
    dist_eq  = np.minimum(np.abs(inc_deg), np.abs(180.0 - inc_deg))
    mask_eq  = (dist_eq < _EPS_I_PLOT_DEG)

    if e_series is None:
        mask_circ = np.zeros_like(inc_deg, dtype=bool)
    else:
        e_s = np.asarray(e_series, float)
        mask_circ = (e_s < e_eps)

    # 4) ZOH/NaN conforme opção
    if use_zoh_on_equatorial:
        Om_cont = _zoh_when_masked(Om_cont, mask_eq)
        w_cont  = _zoh_when_masked(w_cont,  mask_eq)
    else:
        Om_cont = _mask_if(mask_eq, Om_cont)
        w_cont  = _mask_if(mask_eq, w_cont)

    if use_zoh_on_circular:
        w_cont = _zoh_when_masked(w_cont, mask_circ)
    else:
        w_cont = _mask_if(mask_circ, w_cont)

    # 5) volta para [0,360) sem degrau na borda, se pedido
    if show_mod360:
        Om_plot = _to_0_360_no_edge_jump(Om_cont)
        w_plot  = _to_0_360_no_edge_jump(w_cont)
        nu_plot = _to_0_360_no_edge_jump(nu_cont)
    else:
        Om_plot, w_plot, nu_plot = Om_cont, w_cont, nu_cont

    return Om_plot, w_plot, nu_plot


    return Om_plot, w_plot, nu_plot

def _process_e_for_plot(e):
    """
    Suaviza e e faz snap-to-zero abaixo do limiar (remove tremulação numérica),
    sem “matar” a evolução de ω quando e é pequeno mas não ~0.
    """
    e = np.asarray(e, float)
    N = len(e)
    win_e = max(5, (N // _WIN_E_SMOOTH_F) | 1)
    e_smooth = _rolling_mean(e, win_e)
    return np.where(e_smooth < _EPS_E_PLOT, 0.0, e_smooth)

def _plot_wrapped(ax, t, y_deg, **plot_kw):
    """
    Plota ângulo em [0,360) sem conectar segmentos através do wrap.
    Quebra onde |Δy|>180°.
    """
    t = np.asarray(t, float)
    y = np.asarray(y_deg, float)
    if y.size == 0:
        return
    dy = np.diff(y)
    cuts = np.where(np.abs(dy) > 180.0)[0]
    start = 0
    for c in cuts:
        ax.plot(t[start:c+1], y[start:c+1], **plot_kw)
        start = c + 1
    ax.plot(t[start:], y[start:], **plot_kw)

def plot_classic_orbital_elements(t: np.typing.NDArray, orbital_elementss: list[OrbitalElements]):
    a, e, inc, Om, w, nu = _series_from(orbital_elementss)
    e_plot = _process_e_for_plot(e)
    Om_plot, w_plot, nu_plot = _process_angles_for_plot(
        inc, Om, w, nu,
        show_mod360=True,              # exibe 0–360° sem degrau
        use_zoh_on_equatorial=True,
        use_zoh_on_circular=True,
        e_series=e,
        e_eps=_EPS_E_PLOT
    )
    # --- Argumento de latitude: u = ν + ω ---
    # usa pipeline idêntico: contiguous -> suavização -> ZOH em equatorial
    u_raw = (w + nu) % 360.0
    N = len(u_raw)
    win_ang = max(5, (N // _WIN_ANG_SMOOTH_F) | 1)

    u_cont = _contiguous_from_prev(u_raw)
    u_cont = _rolling_mean(u_cont, win_ang)

    # <<< NOVO: traz de volta para [0, 360) sem “dente” na borda
    u_mod  = _to_0_360_no_edge_jump(u_cont)

    dist_eq = np.minimum(np.abs(inc), np.abs(180.0 - inc))
    mask_eq = (dist_eq < _EPS_I_PLOT_DEG)

    # mantém a convenção: em trechos ~equatoriais u é indefinido → ZOH
    u_plot = _zoh_when_masked(u_mod, mask_eq)


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

# ===== RAAN (Ω) — sem processamento; apenas escala ampla para “achatar” ruído =====
  # ===== RAAN (Ω) — sem saltos 0↔360 (unwrap simples relativo ao início) =====
    Om_raw  = np.array([el.ascending_node for el in orbital_elementss], dtype=float)

    # desenrola a série para manter continuidade local (remove saltos de 360°)
    Om_cont = np.empty_like(Om_raw)
    Om_cont[0] = Om_raw[0]
    for k in range(1, Om_raw.size):
        delta = ((Om_raw[k] - Om_raw[k-1] + 180.0) % 360.0) - 180.0
        Om_cont[k] = Om_cont[k-1] + delta

    # plota ΔΩ = Ω(t) − Ω(0) (fica contínuo e centrado em 0)
    Om_rel = Om_cont - Om_cont[0]

    axs[1, 1].plot(t, Om_rel, label='RAAN (ΔΩ contínuo)')
    axs[1, 1].axhline(0.0, color='k', lw=0.8, alpha=0.4)
    axs[1, 1].set_title('Ascending Node (RAAN)')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('ΔΩ (deg)') 
    Om0 = float(Om_raw[0])
    WINDOW = 5.0e-3 # ±0.0001 deg
    axs[1, 1].set_ylim(Om0 - WINDOW, Om0 + WINDOW)

    # ticks “bonitos”
    axs[1, 1].set_yticks(np.linspace(Om0 - WINDOW, Om0 + WINDOW, 5))
    axs[1, 1].ticklabel_format(axis='y', style='plain', useOffset=False)
    axs[1, 1].grid(True)
    axs[1, 1].legend()


    u_series = [(el.true_anomaly + el.argument_of_perigee) % 360.0 for el in orbital_elementss]

    axs[2, 0].plot(t, u_series, label='Argument of Latitude (u)')
    axs[2, 0].set_title('Argument of Latitude (u)')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylabel('Argument of Latitude (degrees)')
    axs[2, 0].set_ylim(0, 360)
    axs[2, 0].set_yticks([0, 60, 120, 180, 240, 300, 360])
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
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    for idx, orbital_elementss_list in enumerate(orbital_elementss_lists):
        t = orbital_elementss_list[0]
        orbital_elementss = orbital_elementss_list[1]

        a, e, inc, Om, w, nu = _series_from(orbital_elementss)
        e_plot = _process_e_for_plot(e)
        Om_plot, w_plot, nu_plot = _process_angles_for_plot(
            inc, Om, w, nu,
            show_mod360=True,          # exibe 0–360° sem degrau
            use_zoh_on_equatorial=True,
            use_zoh_on_circular=True,
            e_series=e,
            e_eps=_EPS_E_PLOT
        )
        # --- Argumento de latitude: u = ν + ω ---
        u_raw = (w + nu) % 360.0
        N = len(u_raw)
        win_ang = max(5, (N // _WIN_ANG_SMOOTH_F) | 1)

        u_cont = _contiguous_from_prev(u_raw)
        u_cont = _rolling_mean(u_cont, win_ang)
        u_mod  = _to_0_360_no_edge_jump(u_cont)   # <<< NOVO
        dist_eq = np.minimum(np.abs(inc), np.abs(180.0 - inc))
        mask_eq = (dist_eq < _EPS_I_PLOT_DEG)
        u_plot = _zoh_when_masked(u_mod, mask_eq)


        axs[0, 0].plot(t, a, label=f'Major Axis #{idx+1}')
        axs[0, 1].plot(t, e_plot, label=f'Eccentricity #{idx+1}')
        axs[1, 0].plot(t, inc, label=f'Inclination #{idx+1}')

        Om_raw  = np.array([el.ascending_node for el in orbital_elementss], dtype=float)
        Om_cont = np.empty_like(Om_raw)
        Om_cont[0] = Om_raw[0]
        for k in range(1, Om_raw.size):
            delta = ((Om_raw[k] - Om_raw[k-1] + 180.0) % 360.0) - 180.0
            Om_cont[k] = Om_cont[k-1] + delta
        Om_rel = Om_cont - Om_cont[0]
        axs[1, 1].plot(t, Om_rel, label=f'RAAN ΔΩ #{idx+1}')

        axs[2, 0].plot(t, u_plot, label=f'Argument of Latitude (u) #{idx+1}')
        axs[2, 1].plot(t, nu_plot, label=f'True Anomaly #{idx+1}')

    axs[0, 0].set_title('Major Axis');      axs[0, 0].set_xlabel('Time (s)'); axs[0, 0].set_ylabel('Major Axis (km)'); axs[0, 0].grid(True); axs[0, 0].legend()
    axs[0, 1].set_title('Eccentricity');    axs[0, 1].set_xlabel('Time (s)'); axs[0, 1].set_ylabel('Eccentricity');     axs[0, 1].grid(True); axs[0, 1].legend()
    axs[1, 0].set_title('Inclination');     axs[1, 0].set_xlabel('Time (s)'); axs[1, 0].set_ylabel('Inclination (deg)');axs[1, 0].grid(True); axs[1, 0].legend()
    axs[1, 1].set_title('Ascending Node');  axs[1, 1].set_xlabel('Time (s)'); axs[1, 1].set_ylabel('Ascending Node (deg)'); axs[1, 1].grid(True); axs[1, 1].legend()
    axs[2, 0].set_title('Argument of Latitude (u)'); axs[2, 0].set_xlabel('Time (s)'); axs[2, 0].set_ylabel('Argument of Latitude (deg)');
    axs[2, 0].set_ylim(0.0, 360.0)                       # <<< novo
    axs[2, 0].set_yticks([0, 60, 120, 180, 240, 300, 360])  # <<< novo
    axs[2, 0].grid(True); axs[2, 0].legend()
    axs[2, 1].set_title('True Anomaly');    axs[2, 1].set_xlabel('Time (s)'); axs[2, 1].set_ylabel('True Anomaly (deg)'); axs[2, 1].grid(True); axs[2, 1].legend()
    axs[1, 1].set_ylim(0.0, 360.0)
    axs[1, 1].set_yticks([0, 60, 120, 180, 240, 300, 360])
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
