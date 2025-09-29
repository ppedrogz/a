import numpy as np
from utils.types import OrbitalElements
from .constants import Bases

# ---------------- utilidades seguras ----------------
def _safe_norm(x, eps=1e-18): 
    n = float(np.linalg.norm(x))
    return n if n > eps else 0.0

def _safe_unit(x, eps=1e-18):
    n = _safe_norm(x, eps)
    return x/n if n > 0.0 else np.zeros_like(x)

def _clamp(x, a=-1.0, b=1.0):
    return a if x < a else (b if x > b else x)

def _safe_acos(x): 
    return float(np.arccos(_clamp(x)))

def _wrap360_deg(th_rad):
    return float(np.degrees(th_rad) % 360.0)

def _atan2(y, x):
    return float(np.arctan2(y, x))

# tolerâncias para detectar circular/equatorial
_EPS_E = 2e-3    
_EPS_I = np.deg2rad(1e-8)

# ---------------- API unificada ----------------
def get_orbital_elements(*args) -> OrbitalElements:
    """
    Aceita:
      - get_orbital_elements(X, mu)        # X = [rx,ry,rz,vx,vy,vz]
      - get_orbital_elements(r, v, mu)
    Retorna OrbitalElements com 'true_anormaly' em [0, 360).
    Trata casos circulares/equatoriais com ifs.
    """
    if len(args) == 2:
        X, mu = args
        r = np.asarray(X[0:3], float).reshape(3)
        v = np.asarray(X[3:6], float).reshape(3)
    elif len(args) == 3:
        r, v, mu = args
        r = np.asarray(r, float).reshape(3)
        v = np.asarray(v, float).reshape(3)
    else:
        raise TypeError("get_orbital_elements: use (X, mu) ou (r, v, mu)")

    a = get_major_axis(r, v, mu)
    e = get_eccentricity(r, v, mu)

    # vetores básicos
    h = np.cross(r, v); hnorm = _safe_norm(h); hhat = _safe_unit(h)
    n = np.cross(Bases.k, h); nnorm = _safe_norm(n)

    # inclinação
    i_rad = _safe_acos(h[2]/hnorm) if hnorm > 0.0 else 0.0
    i_deg = _wrap360_deg(i_rad)

    # nós e ângulos (com ifs p/ singularidades)
    if nnorm > 0.0:
        Omega_rad = _atan2(n[1], n[0])  # RAAN
    else:
        Omega_rad = 0.0  # equatorial → indefinido

    if e > _EPS_E and i_rad > _EPS_I:
        # caso geral (elíptica e inclinada)
        e_vec = get_eccentricity_vector(r, v, mu)
        omega_rad = _atan2(np.dot(np.cross(n, e_vec), hhat), np.dot(n, e_vec))
        nu_rad    = _atan2(np.dot(np.cross(e_vec, r), hhat), np.dot(e_vec, r))
        omega_deg = _wrap360_deg(omega_rad)
        nu_deg    = _wrap360_deg(nu_rad)
        Omega_deg = _wrap360_deg(Omega_rad)
    elif e <= _EPS_E and i_rad > _EPS_I:
        # circular-inclinada → usar u (argumento de latitude)
        # u = atan2((n×r)·ĥ, n·r)
        u_rad     = _atan2(np.dot(np.cross(n, r), hhat), np.dot(n, r)) if nnorm > 0.0 else 0.0
        Omega_deg = _wrap360_deg(Omega_rad)
        omega_deg = 0.0                # indefinido → 0
        nu_deg    = _wrap360_deg(u_rad)  # carregamos a posição orbital em 'ν'
    elif e > _EPS_E and i_rad <= _EPS_I:
        # elíptica-equatorial → usar varpi (Ω+ω) e fixar Ω=0
        e_vec     = get_eccentricity_vector(r, v, mu)
        varpi_rad = _atan2(e_vec[1], e_vec[0])
        nu_rad    = _atan2(np.dot(np.cross(e_vec, r), hhat), np.dot(e_vec, r))
        Omega_deg = 0.0
        omega_deg = _wrap360_deg(varpi_rad)  # info do periastro fica toda aqui
        nu_deg    = _wrap360_deg(nu_rad)
    else:
        # circular-equatorial → usar λ (longitude verdadeira) e fixar Ω=ω=0
        lambda_rad = _atan2(r[1], r[0])
        Omega_deg  = 0.0
        omega_deg  = 0.0
        nu_deg     = _wrap360_deg(lambda_rad)

    return OrbitalElements(
        major_axis=a,
        eccentricity=e,
        inclination=i_deg,
        ascending_node=Omega_deg,
        argument_of_perigee=omega_deg,
        true_anomaly=nu_deg  # mantém o nome que seu código já espera
    )

# ---------------- funções auxiliares (assinaturas antigas mantidas) ----------------
def get_major_axis(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    r_norm = _safe_norm(r)
    v_norm = _safe_norm(v)
    eps = 0.5*(v_norm**2) - (mu/r_norm)
    return -mu/(2.0*eps) if eps != 0.0 else np.inf

def get_eccentricity(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    return _safe_norm(get_eccentricity_vector(r, v, mu))

def get_inclination(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    # atan2(sin(i), cos(i)), where sin(i) = sqrt(hx^2 + hy^2)/|h|, cos(i) = hz/|h|
    sin_i = np.sqrt(h[0]**2 + h[1]**2) / h_norm
    cos_i = h[2] / h_norm

    i_rad = np.arctan2(sin_i, cos_i)

    i = np.rad2deg(i_rad)

    if i < 1e-4:
        i = 0

    return i


def get_ascending_node(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v)
    n = np.cross(Bases.k, h)

    if np.linalg.norm(n) > 1e-10:
        Omega_rad = np.acos(np.dot(Bases.i, n)/np.linalg.norm(n))
        if n[1] < 0:
            Omega_rad = 2*np.pi - Omega_rad
    else:
        Omega_rad = 0

    Omega = np.rad2deg(Omega_rad)

    return Omega

def get_argument_of_perigee(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    # se e ~ 0, ω é indefinido → fixe 0
    e_vec = get_eccentricity_vector(r, v, mu)
    e = _safe_norm(e_vec)
    if e <= _EPS_E:
        return 0.0
    h = np.cross(r, v); hhat = _safe_unit(h)
    n = np.cross(Bases.k, h); nnorm = _safe_norm(n)
    if nnorm <= 0.0:
        # equatorial: ω não é definido separadamente (varpi=Ω+ω). Retorne 0 aqui.
        return 0.0
    omega_rad = _atan2(np.dot(np.cross(n, e_vec), hhat), np.dot(n, e_vec))
    return _wrap360_deg(omega_rad)

def get_true_anormaly(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    """
    Retorna sempre um ângulo de posição em [0,360):
      - caso geral: ν
      - circular-inclinada: u (substitui ν)
      - elíptica-equatorial: ν (regular)
      - circular-equatorial: λ
    """
    h = np.cross(r, v); hhat = _safe_unit(h)
    n = np.cross(Bases.k, h); nnorm = _safe_norm(n)
    e_vec = get_eccentricity_vector(r, v, mu)
    e = _safe_norm(e_vec)

    if e > _EPS_E:
        # ν = atan2((e×r)·ĥ, e·r)
        nu = _atan2(np.dot(np.cross(e_vec, r), hhat), np.dot(e_vec, r))
        return _wrap360_deg(nu)

    # e ~ 0
    if nnorm > 0.0:
        # circular-inclinada → u
        u = _atan2(np.dot(np.cross(n, r), hhat), np.dot(n, r))
        return _wrap360_deg(u)
    else:
        # circular-equatorial → λ
        lam = _atan2(r[1], r[0])
        return _wrap360_deg(lam)

def get_eccentricity_vector(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> np.typing.NDArray:
    h = np.cross(r, v)
    return (np.cross(v, h)/mu) - (r/(_safe_norm(r)+1e-32))

def get_period(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    a = get_major_axis(r, v, mu)
    return 2.0*np.pi*np.sqrt(abs(a**3)/mu) if np.isfinite(a) else np.inf

# KEPLERIAN ELEMENTS
def get_eccentric_anomaly(theta, e) -> float:
    E_sin = np.sqrt(1-e**2)*np.sin(theta)
    E_cos = 1 + e*np.cos(theta)

    E = np.atan2(E_sin, E_cos)

    return E


def get_mean_angular_motion(period, mu: float) -> float:
    n = 2*np.pi/period

    return n


def get_mean_anomaly(theta, e, mu: float) -> float:
    E = get_eccentric_anomaly(theta, e)
    M = E - e*np.sin(E)

    return M


def get_analitical_time(theta: float, e: float, period: float, t0: float, mu: float) -> float:
    M = get_mean_anomaly(theta, e, mu)
    n = get_mean_angular_motion(period, mu)

    t = t0 + ((M)/n)

    return t
