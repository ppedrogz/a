# utils/visualization.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from utils.angles import contiguify_from_prev, rolling_mean

@dataclass
class ElementsSeries:
    a: np.ndarray
    e: np.ndarray
    i_deg: np.ndarray
    Omega_deg: np.ndarray
    omega_deg: np.ndarray
    nu_deg: np.ndarray
    u_deg: np.ndarray
    ltrue_deg: np.ndarray
    energy: np.ndarray  # compatibilidade

def _process_angles(inc_deg, Om_deg, w_deg, nu_deg, u_deg, ltrue_deg,
                    e_series, win_frac: int = 200):
    N = len(inc_deg)
    win = max(5, (N // win_frac) | 1)

    def cont_smooth(x_deg, do_mod=True):
        x = contiguify_from_prev(np.deg2rad(x_deg))
        x = rolling_mean(x, win)
        x = np.rad2deg(x)
        return np.mod(x, 360.0) if do_mod else x

    Om_cont = cont_smooth(Om_deg, do_mod=False)
    w_cont  = cont_smooth(w_deg, do_mod=False)
    lt_cont = cont_smooth(ltrue_deg, do_mod=False)

    w_mod   = cont_smooth(w_deg, do_mod=True)
    nu_mod  = cont_smooth(nu_deg, do_mod=True)
    u_mod   = cont_smooth(u_deg, do_mod=True)
    lt_mod  = cont_smooth(ltrue_deg, do_mod=True)

    e_eps  = 1e-5
    i_eps  = 1e-3
    mask_circ = (e_series < e_eps)
    dist_eq   = np.minimum(inc_deg, 180.0 - inc_deg)
    mask_eq   = (dist_eq < i_eps)

    Om_pref_cont = np.where(mask_eq, lt_cont - w_cont, Om_cont)
    nu_pref_mod  = np.where(mask_circ, u_mod, nu_mod)

    return Om_pref_cont, w_mod, nu_pref_mod, u_mod, lt_mod


def plot_classic_orbital_elements(t: np.ndarray, elems: ElementsSeries):
    """
    Plota (3x2):
      [0,0] a (km)            [0,1] e
      [1,0] i (deg)           [1,1] Ω (deg)
      [2,0] ω (deg)           [2,1] ν e ω (deg)
    """
    Om_s, w_s, nu_pref, u_s, lt_s = _process_angles(
        elems.i_deg, elems.Omega_deg, elems.omega_deg,
        elems.nu_deg, elems.u_deg, elems.ltrue_deg, elems.e
    )

    # ajuste de escala mais adequada para visualização
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # maior figura
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Linha 0: a e e
    #axs[0, 0].plot(t, elems.a, color="blue", lw=1.8)
    #axs[0, 0].set_title('Semi-eixo maior a [km]')
    #axs[0, 1].plot(t, elems.e, color="orange", lw=1.8)
   # axs[0, 1].set_title('Excentricidade e')
  #  axs[0, 0].set_ylim(min(elems.a)*0.999, max(elems.a)*1.001)
    #axs[0, 1].set_ylim(-0.01, 0.01)

    # Linha 1: i e RAAN
    axs[0, 0].plot(t, elems.i_deg, color="green", lw=1.8)
    axs[0, 0].set_title('Inclinação i [deg]')
   # axs[1, 0].set_ylim(min(elems.i_deg)*0.99, max(elems.i_deg)*1.01)

    axs[0, 1].plot(t, Om_s, color="red", lw=1.8)
    axs[0, 1].set_title('RAAN Ω [deg]')
    #axs[1, 1].set_ylim(min(Om_s)*0.99, max(Om_s)*1.01)

    # Linha 2: ω e ν (último gráfico à direita com legenda explicativa)
   # axs[2, 0].plot(t, u_s, color="cyan", lw=1.8)
    #axs[2, 0].set_title('Argumento de Latitude')
   # axs[2, 0].set_ylim(0, 360)

    axs[1, 0].plot(t, nu_pref, color="maroon", lw=1.8, label='Anomalia Verdadeira ν')
    axs[1, 0].set_title('Anomalia Verdadeira [deg]')
    axs[1, 1].plot(t, w_s, color="purple", lw=1.3, linestyle='--', label='Argumento do Perigeu ω')
    axs[1, 1].set_title('Argumento do Perigeu [deg]')
    #axs[2, 1].legend(loc='best', fontsize=9)
    #axs[2, 1].set_ylim(min(min(nu_pref), min(w_s))*0.99,
                       #max(max(nu_pref), max(w_s))*1.01)

    for ax in axs.ravel():
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel('Tempo [s]')
        ax.set_xlim(t[0], t[-1])

    fig.suptitle("Evolução dos Elementos Orbitais Clássicos Sat_VH_D", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig, axs
