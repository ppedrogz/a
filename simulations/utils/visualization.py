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
    energy: np.ndarray  # mantido para compatibilidade, mesmo sem plotar aqui

def _process_angles(inc_deg, Om_deg, w_deg, nu_deg, u_deg, ltrue_deg,
                    e_series, win_frac: int = 200):
    """
    - Desembrulha e suaviza séries angulares.
    - RAAN: **sem módulo 360°** (contínuo) para evitar serrilhado.
    - Em i≈0: usa Ω_pref ≈ ℓ_true − ω (também contínuo, sem módulo).
    - ν_pref: usa u quando e≈0; caso contrário, ν (ambos com módulo p/ ficar em [0,360) no painel dedicado).
    Retorna: Om_pref_cont, w_s_mod, nu_pref_mod, u_s_mod, lt_s_mod
    """
    N = len(inc_deg)
    win = max(5, (N // win_frac) | 1)

    def cont_smooth(x_deg, do_mod=True):
        x = contiguify_from_prev(np.deg2rad(x_deg))
        x = rolling_mean(x, win)
        x = np.rad2deg(x)
        return np.mod(x, 360.0) if do_mod else x

    # Séries contínuas (sem módulo) p/ Ω, ω e ℓ_true
    Om_cont = cont_smooth(Om_deg,   do_mod=False)
    w_cont  = cont_smooth(w_deg,    do_mod=False)
    lt_cont = cont_smooth(ltrue_deg,do_mod=False)

    # Séries com módulo p/ painéis que devem ficar em [0,360)
    w_mod   = cont_smooth(w_deg,    do_mod=True)
    nu_mod  = cont_smooth(nu_deg,   do_mod=True)
    u_mod   = cont_smooth(u_deg,    do_mod=True)
    lt_mod  = cont_smooth(ltrue_deg,do_mod=True)

    # Máscaras
    e_eps  = 1e-5
    i_eps  = 1e-3  # graus
    mask_circ = (e_series < e_eps)
    dist_eq   = np.minimum(inc_deg, 180.0 - inc_deg)
    mask_eq   = (dist_eq < i_eps)

    # i≈0 -> prefira Ω ≈ ℓ_true − ω (tudo contínuo); caso contrário, use Ω contínuo
    Om_pref_cont = np.where(mask_eq, lt_cont - w_cont, Om_cont)

    # e≈0 -> prefira u; caso contrário, ν (com módulo para o painel em [0,360))
    nu_pref_mod = np.where(mask_circ, u_mod, nu_mod)

    return Om_pref_cont, w_mod, nu_pref_mod, u_mod, lt_mod


def plot_classic_orbital_elements(t: np.ndarray, elems: ElementsSeries):
    """
    Plota (3x2):
      [0,0] a (km)            [0,1] e
      [1,0] i (deg)           [1,1] Ω preferencial (deg)
      [2,0] u (arg. latitude) [2,1] ν_pref (ν ou u quando e≈0)
    """
    Om_s, w_s, nu_pref, u_s, lt_s = _process_angles(
        elems.i_deg, elems.Omega_deg, elems.omega_deg, elems.nu_deg,
        elems.u_deg, elems.ltrue_deg, elems.e
    )

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # Linha 0: a (azul) e e (amarelo)
    axs[0, 0].plot(t, elems.a, color="blue")
    axs[0, 0].set_title('Semi-eixo maior a [km]')
    axs[0, 1].plot(t, elems.e, color="yellow")
    axs[0, 1].set_title('Excentricidade e')
   # axs[0, 1].set_ylim(-1, 1)
    # Linha 1: i (verde) e RAAN (vermelho, contínuo/corrigido)
    axs[1, 0].plot(t, elems.i_deg, color="green")
    axs[1, 0].set_title('Inclinação i [deg]')
    axs[1, 1].plot(t, Om_s, color="red")
    axs[1, 1].set_title('RAAN Ω [deg]')
    axs[1, 1].set_ylim(-10, 10)

    # Linha 2: u (roxo) e ν_pref (vinho/maroon)
    axs[2, 0].plot(t, u_s, color="cyan")
    axs[2, 0].set_title('Argumento de latitude [deg]')
    axs[2, 1].plot(t, nu_pref, color="maroon")
    axs[2, 1].plot(t, w_s, color="purple")
    axs[2, 1].set_title('Anomalia verdadeira com Argumento de Perigeu [deg]')

    for ax in axs.ravel():
        ax.grid(True)
        ax.set_xlabel('Tempo [s]')
    fig.tight_layout()
    return fig, axs
