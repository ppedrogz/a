import numpy as np
from utils.types import OrbitalElements
from .constants import Bases

# ---------------- utilidades numéricas seguras ----------------
def _safe_norm(x, eps=1e-18):
    n = float(np.linalg.norm(x))
    return n if n > eps else 0.0

def _safe_unit(x, eps=1e-18):
    n = _safe_norm(x, eps)
    return x/n if n > 0.0 else np.zeros_like(x)

def _clamp(x, a=-1.0, b=1.0):
    return a if x < a else (b if x > b else x)

def _wrap360_deg(theta_rad: float) -> float:
    return float(np.degrees(theta_rad) % 360.0)

def _atan2(y: float, x: float) -> float:
    return float(np.arctan2(y, x))

# tolerâncias práticas (ajuste se quiser “chamar de circular” mais cedo)
_EPS_E = 2e-3              # ~0.002: trata e muito pequeno como circular p/ ângulos
_EPS_I = np.deg2rad(1e-8)  # ~1e-8 deg: equatorial p/ ângulos

# ---------------- API principal ----------------
def get_orbital_elements(X: np.typing.NDArray, mu: float) -> OrbitalElements:
    """
    Retorna SEMPRE os elementos clássicos:
      - ascending_node (Ω) e argument_of_perigee (ω) são fixados em 0 quando indefinidos;
      - true_anomaly (ν) sempre carrega a posição angular 'correta':
        caso geral: ν; circular-inclinada: u; circular-equatorial: λ.
    """
    r = np.asarray(X[0:3], float).reshape(3)
    v = np.asarray(X[3:6], float).reshape(3)

    a  = get_major_axis(r, v, mu)
    e  = get_eccentricity(r, v, mu)
    inc_deg = get_inclination(r, v, mu)

    # vetores básicos
    h = np.cross(r, v); hnorm = _safe_norm(h); hhat = _safe_unit(h)
    n = np.cross(Bases.k, h); nnorm = _safe_norm(n)

    # RAAN (Ω) robusto
    Omega_deg = _wrap360_deg(_atan2(n[1], n[0])) if nnorm > 0.0 else 0.0

    # classificação só para lógica interna
    i_rad = np.deg2rad(inc_deg)
    circular   = (e <= _EPS_E)
    equatorial = (i_rad <= _EPS_I) or (np.abs(inc_deg - 180.0) <= np.rad2deg(_EPS_I))

    if not circular and not equatorial:
        # ----- caso geral (elíptica e inclinada): Ω, ω, ν clássicos -----
        e_vec = get_eccentricity_vector(r, v, mu)
        omega_deg = _wrap360_deg(_atan2(np.dot(np.cross(n, e_vec), hhat), np.dot(n, e_vec))) if nnorm > 0.0 else 0.0
        nu_deg    = _wrap360_deg(_atan2(np.dot(np.cross(e_vec, r), hhat), np.dot(e_vec, r)))
        return OrbitalElements(
            major_axis=a,
            eccentricity=e,
            inclination=inc_deg,
            ascending_node=Omega_deg,
            argument_of_perigee=omega_deg,
            true_anomaly=nu_deg
        )

    if circular and not equatorial:
        # ----- circular-inclinada: usar u; Ω definido; ω indefinido -----
        # u = atan2((n×r)·ĥ , n·r)
        u_rad = _atan2(np.dot(np.cross(n, r), hhat), np.dot(n, r)) if nnorm > 0.0 else 0.0
        return OrbitalElements(
            major_axis=a,
            eccentricity=0.0,
            inclination=inc_deg,
            ascending_node=Omega_deg,
            argument_of_perigee=0.0,                 # indefinido
            true_anomaly=_wrap360_deg(u_rad)         # posição no plano orbital
        )

    if (not circular) and equatorial:
        # ----- elíptica-equatorial: usar ϖ=Ω+ω; Ω indefinido (fixa 0); ν regular -----
        e_vec  = get_eccentricity_vector(r, v, mu)
        varpi_deg = _wrap360_deg(_atan2(e_vec[1], e_vec[0]))
        nu_deg    = _wrap360_deg(_atan2(np.dot(np.cross(e_vec, r), hhat), np.dot(e_vec, r)))
        return OrbitalElements(
            major_axis=a,
            eccentricity=e,
            inclination=0.0 if inc_deg < 90.0 else 180.0,  # só para limpeza
            ascending_node=0.0,                             # indefinido
            argument_of_perigee=varpi_deg,                  # guarda ϖ aqui
            true_anomaly=nu_deg
        )

    # ----- circular-equatorial: usar λ; Ω e ω indefinidos -----
    lambda_deg = _wrap360_deg(_atan2(r[1], r[0]))
    return OrbitalElements(
        major_axis=a,
        eccentricity=0.0,
        inclination=0.0 if inc_deg < 90.0 else 180.0,
        ascending_node=0.0,
        argument_of_perigee=0.0,
        true_anomaly=lambda_deg
    )

# ---------------- funções auxiliares consistentes ----------------
def get_major_axis(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    r_norm = _safe_norm(r)
    v_norm = _safe_norm(v)
    eps = 0.5*(v_norm**2) - (mu/r_norm)
    return -mu/(2.0*eps) if eps != 0.0 else np.inf

def get_eccentricity_vector(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> np.typing.NDArray:
    h = np.cross(r, v)
    return (np.cross(v, h)/mu) - (r/(_safe_norm(r) + 1e-32))

def get_eccentricity(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e = _safe_norm(get_eccentricity_vector(r, v, mu))
    # snap para zero se muito pequeno (para estabilidade dos ifs)
    return 0.0 if e < _EPS_E else e

def get_inclination(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v)
    h_norm = _safe_norm(h)
    cos_i = h[2]/h_norm if h_norm > 0.0 else 1.0
    # clamp + arccos
    i_rad = np.arccos(_clamp(cos_i))
    i_deg = np.degrees(i_rad)
    # snap para 0 ou 180 se muito perto (limpa ruído)
    if i_deg < 1e-6:
        return 0.0
    if abs(i_deg - 180.0) < 1e-6:
        return 180.0
    return i_deg

def get_ascending_node(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v); n = np.cross(Bases.k, h)
    if _safe_norm(n) <= 0.0:
        return 0.0
    return _wrap360_deg(_atan2(n[1], n[0]))

def get_argument_of_perigee(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e_vec = get_eccentricity_vector(r, v, mu); e = _safe_norm(e_vec)
    if e <= _EPS_E:
        return 0.0  # circular → indefinido
    h = np.cross(r, v); hhat = _safe_unit(h)
    n = np.cross(Bases.k, h); nnorm = _safe_norm(n)
    if nnorm <= 0.0:
        return 0.0  # equatorial → indefinido (usar ϖ)
    omega_rad = _atan2(np.dot(np.cross(n, e_vec), hhat), np.dot(n, e_vec))
    return _wrap360_deg(omega_rad)

def get_true_anomaly(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    """
    Retorna sempre o ângulo de posição que faz sentido:
      - geral: ν;
      - circular-inclinada: u (argumento de latitude);
      - circular-equatorial: λ (longitude verdadeira).
    """
    h = np.cross(r, v); hhat = _safe_unit(h)
    n = np.cross(Bases.k, h); nnorm = _safe_norm(n)
    e_vec = get_eccentricity_vector(r, v, mu); e = _safe_norm(e_vec)

    if e > _EPS_E:
        nu = _atan2(np.dot(np.cross(e_vec, r), hhat), np.dot(e_vec, r))
        return _wrap360_deg(nu)

    if nnorm > 0.0:
        # u = atan2((n×r)·ĥ, n·r)
        u = _atan2(np.dot(np.cross(n, r), hhat), np.dot(n, r))
        return _wrap360_deg(u)

    # circular-equatorial → λ
    lam = _atan2(r[1], r[0])
    return _wrap360_deg(lam)

def get_period(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    a = get_major_axis(r, v, mu)
    return 2.0*np.pi*np.sqrt(abs(a**3)/mu) if np.isfinite(a) else np.inf

# --------- elementos "alternativos" (se você precisar individualmente) ---------
def get_argument_of_latitude(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v); hhat = _safe_unit(h)
    n = np.cross(Bases.k, h); nnorm = _safe_norm(n)
    if nnorm <= 0.0:
        # equatorial → use λ
        return _wrap360_deg(_atan2(r[1], r[0]))
    u = _atan2(np.dot(np.cross(n, r), hhat), np.dot(n, r))
    return _wrap360_deg(u)

def get_longitude_of_periapsis(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e_vec = get_eccentricity_vector(r, v, mu); e = _safe_norm(e_vec)
    if e <= _EPS_E:
        return 0.0
    # ϖ = atan2(e_y, e_x)
    return _wrap360_deg(_atan2(e_vec[1], e_vec[0]))

def get_true_longitude(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    # λ = atan2(y, x)
    return _wrap360_deg(_atan2(r[1], r[0]))

# --------- Kepler auxiliares (mantidos) ---------
def get_eccentric_anomaly(theta_rad, e) -> float:
    E_sin = np.sqrt(1-e**2)*np.sin(theta_rad)
    E_cos = 1 + e*np.cos(theta_rad)
    return float(np.arctan2(E_sin, E_cos))

def get_mean_angular_motion(period, mu: float) -> float:
    return float(2*np.pi/period)

def get_mean_anomaly(theta_rad, e, mu: float) -> float:
    E = get_eccentric_anomaly(theta_rad, e)
    return float(E - e*np.sin(E))

def get_analitical_time(theta: float, e: float, period: float, t0: float, mu: float) -> float:
    M = get_mean_anomaly(theta, e, mu)
    n = get_mean_angular_motion(period, mu)
    return float(t0 + (M)/n)
