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

# ---------------- parâmetros “de plot” (não afetam dinâmica) ---------------
# pode ajustar conforme o ruído da tua integração
_EPS_I_PLOT_DEG = 1e-4      # trata i≈0° ou i≈180° como equatorial p/ Ω (indefinido no plot)
_EPS_E_PLOT     = 2e-3      # trata e<0.002 como “quase circular” p/ suavizar excentricidade
_WIN_E_SMOOTH_F = 200       # janela ~N/_WIN_E_SMOOTH_F para e
_WIN_ANG_SMOOTH_F = 300     # janela ~N/_WIN_ANG_SMOOTH_F para ângulos

def _series_from(orbital_elementss: list[OrbitalElements]):
    a  = np.array([e.major_axis           for e in orbital_elementss], float)
    ec = np.array([e.eccentricity         for e in orbital_elementss], float)
    inc= np.array([e.inclination          for e in orbital_elementss], float)     # deg
    Om = np.array([e.ascending_node       for e in orbital_elementss], float)     # deg
    w  = np.array([e.argument_of_perigee  for e in orbital_elementss], float)     # deg
    nu = np.array([e.true_anomaly         for e in orbital_elementss], float)     # deg
    return a, ec, inc, Om, w, nu

def _process_angles_for_plot(inc_deg, Om_deg, w_deg, nu_deg):
    N = len(inc_deg)
    # janelas de suavização proporcionais ao tamanho das séries
    win_ang = max(5, (N // _WIN_ANG_SMOOTH_F) | 1)
    # unwrap + suavização leve + remap 0..360
    Om_cont = _mod360_cont(_rolling_mean(_unwrap_deg(Om_deg), win_ang))
    w_cont  = _mod360_cont(_rolling_mean(_unwrap_deg(w_deg ), win_ang))
    nu_cont = _mod360_cont(_rolling_mean(_unwrap_deg(nu_deg), win_ang))

    # máscara equatorial para Ω e ω: i próximo de 0° OU 180° → indefinidos
    dist_eq = np.minimum(np.abs(inc_deg), np.abs(180.0 - inc_deg))
    mask_eq = (dist_eq < _EPS_I_PLOT_DEG)

    Om_plot = _mask_if(mask_eq, Om_cont)
    w_plot  = _mask_if(mask_eq, w_cont)
    # (ν pode ser definido mesmo em casos alternativos, pois tua rotina já entrega u/λ)
    nu_plot = nu_cont

    return Om_plot, w_plot, nu_plot

def _process_e_for_plot(e):
    N = len(e)
    win_e = max(5, (N // _WIN_E_SMOOTH_F) | 1)
    e_smooth = _rolling_mean(e, win_e)
    # “snap to zero” abaixo do limiar para remover tremulação numérica
    e_plot = np.where(e_smooth < _EPS_E_PLOT, 0.0, e_smooth)
    return e_plot

def plot_classic_orbital_elements(t: np.typing.NDArray, orbital_elementss: list[OrbitalElements]):
    """
    Plota elementos orbitais clássicos com tratamentos de plot para:
    - Ω: unwrap + máscara em órbita equatorial (i≈0° ou 180°)
    - e: suavização e snap-to-zero para quase-circular
    - ω, ν: unwrap para eliminar degraus 360→0
    """
    a, e, inc, Om, w, nu = _series_from(orbital_elementss)
    Om_plot, w_plot, nu_plot = _process_angles_for_plot(inc, Om, w, nu)
    e_plot = _process_e_for_plot(e)

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
        Om_plot, w_plot, nu_plot = _process_angles_for_plot(inc, Om, w, nu)
        e_plot = _process_e_for_plot(e)

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

