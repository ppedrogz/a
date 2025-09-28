import numpy as np
from utils.types import OrbitalElements
from .constants import Bases

# ================= utilidades numéricas seguras =================
def _safe_norm(x, eps=1e-18):
    n = float(np.linalg.norm(x))
    return n if n > eps else 0.0

def _safe_unit(x, eps=1e-18):
    n = _safe_norm(x, eps)
    return x/n if n > 0.0 else np.zeros_like(x)

def _clamp(x, a=-1.0, b=1.0):
    return a if x < a else (b if x > b else x)

def _wrap360_deg(th_rad: float) -> float:
    return float(np.degrees(th_rad) % 360.0)

def _atan2(y: float, x: float) -> float:
    return float(np.arctan2(y, x))

# Tolerâncias práticas para decidir "quase circular/equatorial" no PLOT/elementos
_EPS_E = 0.1              # trata e < 0.002 como circular para os ângulos
_EPS_I = np.deg2rad(1e-8)  # trata i muito pequeno como equatorial

# ================= API principal (aceita X ou (r,v)) =================
def get_orbital_elements(*args) -> OrbitalElements:
    """
    Aceita:
      - get_orbital_elements(X, mu)        # X = [rx,ry,rz,vx,vy,vz]
      - get_orbital_elements(r, v, mu)
    Retorna sempre os elementos clássicos. Em casos degenerados:
      - circular-inclinada: ω := 0 e true_anomaly := u (argumento da latitude)
      - elíptica-equatorial: Ω := 0 e argument_of_perigee := ϖ (longitude do periastro)
      - circular-equatorial: Ω := 0, ω := 0 e true_anomaly := λ (longitude verdadeira)
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

    a   = get_major_axis(r, v, mu)
    e   = get_eccentricity(r, v, mu)  # já com snap-to-zero
    inc = get_inclination(r, v, mu)

    # vetores básicos
    h = np.cross(r, v); hnorm = _safe_norm(h); hhat = _safe_unit(h)
    n = np.cross(Bases.k, h); nnorm = _safe_norm(n)

    # RAAN Ω robusto (se equatorial, 0)
    Omega_deg = _wrap360_deg(_atan2(n[1], n[0])) if nnorm > 0.0 else 0.0

    # flags de regime
    i_rad = np.deg2rad(inc)
    circular   = (e <= _EPS_E)
    equatorial = (i_rad <= _EPS_I) or (abs(inc - 180.0) <= np.rad2deg(_EPS_I))

    if not circular and not equatorial:
        # ----- elíptica & inclinada: Ω, ω, ν clássicos
        e_vec = get_eccentricity_vector(r, v, mu)
        omega_deg = _wrap360_deg(_atan2(np.dot(np.cross(n, e_vec), hhat), np.dot(n, e_vec))) if nnorm > 0.0 else 0.0
        nu_deg    = _wrap360_deg(_atan2(np.dot(np.cross(e_vec, r), hhat), np.dot(e_vec, r)))
        return OrbitalElements(
            major_axis=a, eccentricity=e, inclination=inc,
            ascending_node=Omega_deg, argument_of_perigee=omega_deg, true_anomaly=nu_deg
        )

    if circular and not equatorial:
        # ----- circular-inclinada: usar u em vez de ν; ω indefinido
        # u = atan2((n×r)·ĥ, n·r)
        u_rad = _atan2(np.dot(np.cross(n, r), hhat), np.dot(n, r)) if nnorm > 0.0 else 0.0
        return OrbitalElements(
            major_axis=a, eccentricity=0.0, inclination=inc,
            ascending_node=Omega_deg, argument_of_perigee=0.0, true_anomaly=_wrap360_deg(u_rad)
        )

    if (not circular) and equatorial:
        # ----- elíptica-equatorial: usar ϖ=atan2(e_y, e_x) no lugar de ω; Ω indefinido
        e_vec = get_eccentricity_vector(r, v, mu)
        varpi_deg = _wrap360_deg(_atan2(e_vec[1], e_vec[0]))
        nu_deg    = _wrap360_deg(_atan2(np.dot(np.cross(e_vec, r), hhat), np.dot(e_vec, r)))
        return OrbitalElements(
            major_axis=a, eccentricity=e, inclination=0.0 if inc < 90.0 else 180.0,
            ascending_node=0.0, argument_of_perigee=varpi_deg, true_anomaly=nu_deg
        )

    # ----- circular-equatorial: usar λ=atan2(y,x); Ω e ω indefinidos
    lambda_deg = _wrap360_deg(_atan2(r[1], r[0]))
    return OrbitalElements(
        major_axis=a, eccentricity=0.0, inclination=0.0 if inc < 90.0 else 180.0,
        ascending_node=0.0, argument_of_perigee=0.0, true_anomaly=lambda_deg
    )

# ================= auxiliares consistentes =================
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
    return 0.0 if e < _EPS_E else e  # snap-to-zero para estabilidade

def get_inclination(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v)
    h_norm = _safe_norm(h)
    cos_i = h[2]/h_norm if h_norm > 0.0 else 1.0
    i_rad = np.arccos(_clamp(cos_i))
    i_deg = np.degrees(i_rad)
    # se extremamente perto de 0 ou 180, limpa ruído
    if i_deg < 1e-6: return 0.0
    if abs(i_deg - 180.0) < 1e-6: return 180.0
    return i_deg

def get_ascending_node(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v); n = np.cross(Bases.k, h)
    if _safe_norm(n) <= 0.0:
        return 0.0
    return _wrap360_deg(_atan2(n[1], n[0]))

def get_argument_of_perigee(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e_vec = get_eccentricity_vector(r, v, mu); e = _safe_norm(e_vec)
    if e <= _EPS_E:  # circular → ω indefinido
        return 0.0
    h = np.cross(r, v); hhat = _safe_unit(h)
    n = np.cross(Bases.k, h); nnorm = _safe_norm(n)
    if nnorm <= 0.0:  # equatorial → ω indefinido (usa ϖ)
        return 0.0
    omega_rad = _atan2(np.dot(np.cross(n, e_vec), hhat), np.dot(n, e_vec))
    return _wrap360_deg(omega_rad)

def get_true_anomaly(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    """
    Retorna sempre o ângulo de posição que faz sentido:
      - caso geral (e>EPS): ν = atan2((e×r)·ĥ, e·r)
      - circular-inclinada: u  (argumento da latitude)
      - circular-equatorial: λ  (longitude verdadeira)
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

# ================= elementos alternativos (expostos se precisar) =================
def get_argument_of_latitude(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v); hhat = _safe_unit(h)
    n = np.cross(Bases.k, h); nnorm = _safe_norm(n)
    if nnorm <= 0.0:
        return _wrap360_deg(_atan2(r[1], r[0]))  # λ
    u = _atan2(np.dot(np.cross(n, r), hhat), np.dot(n, r))
    return _wrap360_deg(u)

def get_longitude_of_periapsis(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e_vec = get_eccentricity_vector(r, v, mu); e = _safe_norm(e_vec)
    if e <= _EPS_E:
        return 0.0
    return _wrap360_deg(_atan2(e_vec[1], e_vec[0]))  # ϖ

def get_true_longitude(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    return _wrap360_deg(_atan2(r[1], r[0]))  # λ

# ================= Kepler auxiliares (mantidos) =================
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

# --------- alias para compatibilidade com chamadas antigas ---------
def get_orbital_elements_X_mu(X, mu):
    return get_orbital_elements(X, mu)

def get_orbital_elements_rv_mu(r, v, mu):
    return get_orbital_elements(r, v, mu)

# Se em algum ponto do seu código existir o nome antigo "get_true_anormaly":
get_true_anormaly = get_true_anomaly
