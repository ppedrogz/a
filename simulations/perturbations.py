# simulations/perturbations.py
# Unidades: km, s, km^3/s^2 -> acelerações em km/s^2
# Feito para ser chamado só pelo MAIN: diagnostics que imprimem/plotam variações
# de elementos orbitais. O termo J2 (achatamento) está implementado e pronto
# para ser somado no ODE quando você quiser (accel_j2 / accel_perturbations).

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

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

# ========================= Elementos clássicos (internos, sem dependências) =========================
def elements_from_rv(r: np.ndarray, v: np.ndarray, mu: float) -> Dict[str, float]:
    """
    Calcula elementos orbitais clássicos a partir de r, v (km, km/s).
    Retorna dict com a, e, i, RAAN (Omega), argp (omega), nu (graus).
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    R = np.linalg.norm(r)
    V = np.linalg.norm(v)

    h = np.cross(r, v)
    H = np.linalg.norm(h)
    k_hat = np.array([0.0, 0.0, 1.0])

    # excentricidade
    e_vec = (np.cross(v, h) / mu) - (r / R)
    e = np.linalg.norm(e_vec)

    # energia específica
    eps = 0.5 * V*V - mu / R
    a = np.inf if abs(eps) < 1e-30 else -mu / (2.0 * eps)

    # inclinação
    i = np.arccos(np.clip(h[2] / (H + 1e-32), -1.0, 1.0))

    # nó ascendente
    n_vec = np.cross(k_hat, h)
    N = np.linalg.norm(n_vec)
    Omega = np.arctan2(n_vec[1], n_vec[0])  # [-pi, pi]
    if Omega < 0.0:
        Omega += 2.0 * np.pi

    # argumento do perigeu
    if e > 1e-12 and N > 1e-12:
        cos_argp = np.dot(n_vec, e_vec) / (N * e)
        cos_argp = np.clip(cos_argp, -1.0, 1.0)
        argp = np.arccos(cos_argp)
        if e_vec[2] < 0.0:
            argp = 2.0 * np.pi - argp
    else:
        argp = 0.0

    # anomalia verdadeira (0–360 com mapeamento pelo sinal de r·v)
    if e > 1e-12:
        cos_nu = np.dot(e_vec, r) / (e * R)
        cos_nu = np.clip(cos_nu, -1.0, 1.0)
        nu = np.arccos(cos_nu)
        if np.dot(r, v) < 0.0:
            nu = 2.0 * np.pi - nu
    else:
        # órbita circular: define nu pelo ângulo com a direção de linha dos nós
        u_hat = n_vec / (N + 1e-32)
        cos_u = np.dot(u_hat, r / (R + 1e-32))
        cos_u = np.clip(cos_u, -1.0, 1.0)
        nu = np.arccos(cos_u)
        if r[2] < 0.0:
            nu = 2.0 * np.pi - nu

    return dict(
        a=float(a),
        e=float(e),
        i=float(np.degrees(i)),
        Omega=float(np.degrees(Omega)),
        argp=float(np.degrees(argp)),
        nu=float(np.degrees(nu))
    )

# ========================= Diagnósticos para chamar no MAIN =========================
def diagnose_orbital_changes(t: np.ndarray,
                             X: np.ndarray,
                             params: EarthParams = EarthParams(),
                             label: str = "Satélite",
                             print_samples: bool = True) -> Dict[str, float]:
    """
    Lê t (s) e X (6xn ou 7xn) do seu simulate(), computa elementos ao longo do tempo
    e imprime variações relevantes (a, e, i, RAAN, omega). Retorna dict com deltas
    e taxas médias úteis para logs e plots agregados no MAIN.

    Use assim no seu main, depois de cada simulate():
        from simulations.perturbations import diagnose_orbital_changes
        diag_up = diagnose_orbital_changes(t_up, X_up, label="VH UP")
    """
    mu = params.mu

    if X.shape[0] < 6:
        raise ValueError("X deve conter ao menos [r(3); v(3)].")

    r_all = X[0:3, :].T
    v_all = X[3:6, :].T

    elems0 = elements_from_rv(r_all[0], v_all[0], mu)
    elemsF = elements_from_rv(r_all[-1], v_all[-1], mu)

    # séries (opcional para debug/plots)
    if print_samples:
        print(f"\n=== {label}: elementos iniciais ===")
        print(f"a0 [km] = {elems0['a']:.6f} | e0 = {elems0['e']:.9f} | i0 [deg] = {elems0['i']:.6f} | "
              f"RAAN0 [deg] = {elems0['Omega']:.6f} | ω0 [deg] = {elems0['argp']:.6f}")

        print(f"=== {label}: elementos finais ===")
        print(f"aF [km] = {elemsF['a']:.6f} | eF = {elemsF['e']:.9f} | iF [deg] = {elemsF['i']:.6f} | "
              f"RAANF [deg] = {elemsF['Omega']:.6f} | ωF [deg] = {elemsF['argp']:.6f}")

    # deltas e taxas médias
    dt_tot = float(t[-1] - t[0])
    sec_per_day = 86400.0
    days = dt_tot / sec_per_day if dt_tot > 0 else 0.0

    def wrap_deg_diff(xF: float, x0: float) -> float:
        """Diferença angular em graus, embrulhando para [-180, 180]."""
        d = (xF - x0 + 180.0) % 360.0 - 180.0
        return d

    d_a     = elemsF['a']     - elems0['a']
    d_e     = elemsF['e']     - elems0['e']
    d_i     = elemsF['i']     - elems0['i']
    d_RAAN  = wrap_deg_diff(elemsF['Omega'], elems0['Omega'])
    d_argp  = wrap_deg_diff(elemsF['argp'],  elems0['argp'])

    rate_RAAN_deg_day = (d_RAAN / days) if days > 0 else 0.0
    rate_argp_deg_day = (d_argp / days) if days > 0 else 0.0

    print(f"\n--- {label}: variações (final - inicial) ---")
    print(f"Δa [km]   = {d_a:.6f}")
    print(f"Δe        = {d_e:.9f}")
    print(f"Δi [deg]  = {d_i:.9f}")
    print(f"ΔΩ [deg]  = {d_RAAN:.9f}  |  Ω̇ [deg/dia] ≈ {rate_RAAN_deg_day:.9f}")
    print(f"Δω [deg]  = {d_argp:.9f}  |  ω̇ [deg/dia] ≈ {rate_argp_deg_day:.9f}")

    # comparação com taxas seculares J2 (aprox) usando elementos médios (a, e, i)
    a_bar   = 0.5 * (elems0['a'] + elemsF['a'])
    e_bar   = 0.5 * (elems0['e'] + elemsF['e'])
    i_bar   = np.radians(0.5 * (elems0['i'] + elemsF['i']))

    if np.isfinite(a_bar) and a_bar > 0.0:
        n_bar = np.sqrt(mu / (a_bar**3))  # rad/s
        p_bar = a_bar * (1.0 - e_bar*e_bar)
        fac   = -1.5 * params.J2 * (params.Re**2) * n_bar / (p_bar**2)
        # Ω̇ (rad/s) e ω̇ (rad/s)
        raan_dot   = fac * np.cos(i_bar)
        argp_dot   = -0.75 * params.J2 * (params.Re**2) * n_bar / (p_bar**2) * (5.0*np.cos(i_bar)**2 - 1.0)

        # em deg/dia
        r2d = 180.0/np.pi
        raan_dot_deg_day = raan_dot * r2d * sec_per_day
        argp_dot_deg_day = argp_dot * r2d * sec_per_day

        print(f"\n[J2 secular estimado] Ω̇ ≈ {raan_dot_deg_day:.6f} deg/dia | ω̇ ≈ {argp_dot_deg_day:.6f} deg/dia")
        print("(Estimativa por elementos médios; útil para checar tendência dos seus plots.)")

    return dict(
        a0=elems0['a'], aF=elemsF['a'], da=d_a,
        e0=elems0['e'], eF=elemsF['e'], de=d_e,
        i0=elems0['i'], iF=elemsF['i'], di=d_i,
        RAAN0=elems0['Omega'], RAANF=elemsF['Omega'], dRAAN=d_RAAN, RAANdot_deg_day=rate_RAAN_deg_day,
        argp0=elems0['argp'], argpF=elemsF['argp'], dargp=d_argp, argpdot_deg_day=rate_argp_deg_day,
        days=days
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
    "elements_from_rv", "diagnose_orbital_changes",
    "external_accel",
]
