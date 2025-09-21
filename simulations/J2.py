# simulations/perturbations.py
# Unidades: km, s, km^3/s^2 -> acelerações em km/s^2
# Módulo focado em perturbações (J2, por enquanto) e diagnósticos
# para uso a partir do "main", sem alterar os propagadores.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

# --- use seus utilitários existentes de elementos orbitais ---
from utils.GetClassicOrbitalElements import (
    get_orbital_elements,
    get_inclination,
    get_eccentricity_vector,
)

# ========================= Parâmetros e flags =========================
@dataclass(frozen=True)
class EarthParams:
    mu: float = 3.986004418e5  # km^3/s^2
    Re: float = 6378.1363      # km
    J2: float = 1.08262668e-3  # -

@dataclass(frozen=True)
class PerturbationFlags:
    j2: bool = True
    # futuros:
    # drag: bool = False
    # j22: bool = False
    # srp: bool = False
    # third_body: bool = False

# ========================= Acelerações (para quando for usar no ODE) =========================
def accel_twobody(r_vec: np.ndarray, mu: float) -> np.ndarray:
    r = np.linalg.norm(r_vec) + 1e-32
    return -mu * r_vec / (r**3)

def accel_j2(r_vec: np.ndarray, mu: float, Re: float, J2: float) -> np.ndarray:
    """
    Aceleração de achatamento (J2) em frame inercial com eixo z no eixo da Terra.
    Retorna (3,) em km/s^2.
    """
    x, y, z = float(r_vec[0]), float(r_vec[1]), float(r_vec[2])
    r2 = x*x + y*y + z*z
    r  = (r2 + 0.0)**0.5 + 1e-32
    z2 = z*z
    fac = (3.0 * mu * J2 * (Re**2)) / (2.0 * r**5)
    cxy = (5.0 * z2 / r2) - 1.0
    cz  = (5.0 * z2 / r2) - 3.0
    return np.array([fac * x * cxy, fac * y * cxy, fac * z * cz], dtype=float)

def accel_perturbations(r_vec: np.ndarray,
                        v_vec: np.ndarray | None = None,
                        t: float | None = None,
                        params: EarthParams = EarthParams(),
                        flags: PerturbationFlags = PerturbationFlags()) -> np.ndarray:
    """
    Soma das perturbações modeladas. Hoje: somente J2.
    Assinatura preparada para futuros termos (drag, SRP, J22, etc).
    """
    a = np.zeros(3, dtype=float)
    if flags.j2:
        a += accel_j2(r_vec, params.mu, params.Re, params.J2)
    return a

# ========================= True anomaly robusta (sem "dentes") =========================
def true_anomaly_deg_from_rv(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    """
    Anomalia verdadeira 0–360° usando arctan2(sin nu, cos nu), evitando saltos
    359→0 e ambiguidade de quadrante. Implementada com seu get_eccentricity_vector().
    """
    r = np.asarray(r, dtype=float); v = np.asarray(v, dtype=float)
    R = np.linalg.norm(r) + 1e-32

    h = np.cross(r, v); H = np.linalg.norm(h) + 1e-32
    e_vec = get_eccentricity_vector(r, v, mu); e = np.linalg.norm(e_vec) + 1e-32

    # cos(nu) e sin(nu)
    cos_nu = np.dot(e_vec, r) / (e * R)
    cos_nu = np.clip(cos_nu, -1.0, 1.0)
    # sin(nu) = (h · (r × e)) / (H e R)
    sin_nu = np.dot(h, np.cross(r, e_vec)) / (H * e * R)

    nu = np.degrees(np.arctan2(sin_nu, cos_nu))
    if nu < 0.0:
        nu += 360.0
    return float(nu)

def series_nu_i_deg(t: np.ndarray, X: np.ndarray, mu: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Séries de ν (0–360) e i (graus) a partir de X=[r;v;...] ao longo de t.
    ν usa a versão robusta; i usa seu get_inclination().
    """
    r_all = X[0:3, :].T; v_all = X[3:6, :].T
    n = r_all.shape[0]
    nus = np.empty(n, float); incs = np.empty(n, float)
    for k in range(n):
        nus[k] = true_anomaly_deg_from_rv(r_all[k], v_all[k], mu)
        incs[k] = get_inclination(r_all[k], v_all[k], mu)
    return nus, incs

def sort_by_nu_for_plot(nu_deg: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ordena por ν para evitar traços cruzados quando usar linhas."""
    idx = np.argsort(nu_deg)
    return nu_deg[idx], y[idx]

# ========================= Diagnósticos para chamar no MAIN =========================
def diagnose_orbital_changes(t: np.ndarray,
                             X: np.ndarray,
                             params: EarthParams = EarthParams(),
                             label: str = "Satélite",
                             print_samples: bool = True) -> Dict[str, float]:
    """
    Lê t (s) e X (6xn ou 7xn), computa elementos com seus utilitários
    e imprime variações (a, e, i, RAAN, ω) + taxas médias.
    """
    mu = params.mu
    if X.shape[0] < 6:
        raise ValueError("X deve conter ao menos [r(3); v(3)].")

    r_all = X[0:3, :].T
    v_all = X[3:6, :].T

    e0 = get_orbital_elements(r_all[0],   v_all[0],   mu)
    eF = get_orbital_elements(r_all[-1],  v_all[-1],  mu)

    if print_samples:
        print(f"\n=== {label}: elementos iniciais ===")
        print(f"a0 [km] = {e0.major_axis:.6f} | e0 = {e0.eccentricity:.9f} | i0 [deg] = {e0.inclination:.6f} | "
              f"RAAN0 [deg] = {e0.ascending_node:.6f} | ω0 [deg] = {e0.argument_of_perigee:.6f}")
        print(f"=== {label}: elementos finais ===")
        print(f"aF [km] = {eF.major_axis:.6f} | eF = {eF.eccentricity:.9f} | iF [deg] = {eF.inclination:.6f} | "
              f"RAANF [deg] = {eF.ascending_node:.6f} | ωF [deg] = {eF.argument_of_perigee:.6f}")

    # deltas e taxas
    dt_tot = float(t[-1] - t[0]); days = dt_tot/86400.0 if dt_tot>0 else 0.0

    def wrap_deg_diff(xF: float, x0: float) -> float:
        return (xF - x0 + 180.0) % 360.0 - 180.0

    d_a    = eF.major_axis       - e0.major_axis
    d_e    = eF.eccentricity     - e0.eccentricity
    d_i    = eF.inclination      - e0.inclination
    d_RAAN = wrap_deg_diff(eF.ascending_node,     e0.ascending_node)
    d_argp = wrap_deg_diff(eF.argument_of_perigee, e0.argument_of_perigee)

    print(f"\n--- {label}: variações (final - inicial) ---")
    print(f"Δa [km]   = {d_a:.6f}")
    print(f"Δe        = {d_e:.9f}")
    print(f"Δi [deg]  = {d_i:.9f}")
    print(f"ΔΩ [deg]  = {d_RAAN:.9f}  |  Ω̇ [deg/dia] ≈ {(d_RAAN/days if days>0 else 0.0):.9f}")
    print(f"Δω [deg]  = {d_argp:.9f}  |  ω̇ [deg/dia] ≈ {(d_argp/days if days>0 else 0.0):.9f}")

    # estimativa secular J2 (com elementos médios)
    a_bar = 0.5*(e0.major_axis + eF.major_axis)
    e_bar = 0.5*(e0.eccentricity + eF.eccentricity)
    i_bar = np.radians(0.5*(e0.inclination + eF.inclination))
    if np.isfinite(a_bar) and a_bar>0:
        n_bar = np.sqrt(mu/(a_bar**3))
        p_bar = a_bar*(1.0 - e_bar*e_bar)
        fac   = -1.5*params.J2*(params.Re**2)*n_bar/(p_bar**2)
        raan_dot = fac*np.cos(i_bar)
        argp_dot = -0.75*params.J2*(params.Re**2)*n_bar/(p_bar**2)*(5.0*np.cos(i_bar)**2 - 1.0)
        r2d = 180.0/np.pi
        print(f"\n[J2 secular estimado] Ω̇ ≈ {raan_dot*r2d*86400.0:.6f} deg/dia | "
              f"ω̇ ≈ {argp_dot*r2d*86400.0:.6f} deg/dia")

    return dict(
        a0=e0.major_axis, aF=eF.major_axis, da=d_a,
        e0=e0.eccentricity, eF=eF.eccentricity, de=d_e,
        i0=e0.inclination, iF=eF.inclination, di=d_i,
        RAAN0=e0.ascending_node, RAANF=eF.ascending_node, dRAAN=d_RAAN,
        argp0=e0.argument_of_perigee, argpF=eF.argument_of_perigee, dargp=d_argp,
        days=days,
    )

# ========================= Utilitário simples para “hook” futuro =========================
def external_accel(r_vec: np.ndarray, v_vec: np.ndarray, t: float,
                   params: EarthParams = EarthParams(),
                   flags: PerturbationFlags = PerturbationFlags()) -> np.ndarray:
    """
    Callback de perturbações. Se no futuro você decidir somar perturbações direto no ODE,
    basta passar esta função para o integrador (ou injetar via main).
    """
    return accel_perturbations(r_vec, v_vec, t, params=params, flags=flags)

__all__ = [
    "EarthParams", "PerturbationFlags",
    "accel_twobody", "accel_j2", "accel_perturbations",
    "true_anomaly_deg_from_rv", "series_nu_i_deg", "sort_by_nu_for_plot",
    "diagnose_orbital_changes", "external_accel",
]
