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

_EPS_E = 0          # só trata como circular quando e < 0.002
_EPS_I = np.deg2rad(1e-8) # i muito pequeno => equatorial


# ================= helpers de quadro orbital e elementos "circulares" =================
def _orbital_frame(r: np.ndarray, v: np.ndarray):
    """
    Eixos do plano orbital:
      p̂ ao longo da linha dos nós (ascendente) quando definida,
      q̂ = ŵ × p̂,
      ŵ = ĥ.
    """
    h = np.cross(r, v); hhat = _safe_unit(h)
    n = np.cross(Bases.k, h); nnorm = _safe_norm(n)
    if nnorm > 0.0:
        p_hat = n / nnorm
    else:
        # órbita equatorial → defina p̂ = +x por convenção (Ω := 0)
        p_hat = Bases.i
    q_hat = _safe_unit(np.cross(hhat, p_hat))
    return p_hat, q_hat, hhat, nnorm

def _circular_components(r: np.ndarray, v: np.ndarray, mu: float):
    """
    Retorna (e, ex, ey, alpha_v_deg, p̂, q̂, ĥ, nnorm) com:
      ex = e·cos(ω) = e⃗·p̂,  ey = e·sin(ω) = e⃗·q̂,
      αᵥ = ν + ω = atan2(r·q̂, r·p̂).
    """
    e_vec = get_eccentricity_vector(r, v, mu)
    e = _safe_norm(e_vec)
    p_hat, q_hat, hhat, nnorm = _orbital_frame(r, v)
    ex = float(np.dot(e_vec, p_hat))
    ey = float(np.dot(e_vec, q_hat))
    alpha_v = _atan2(np.dot(r, q_hat), np.dot(r, p_hat))
    return e, ex, ey, _wrap360_deg(alpha_v), p_hat, q_hat, hhat, nnorm


# ================= API principal (aceita X ou (r,v)) =================
def get_orbital_elements(*args) -> OrbitalElements:
    """
    Aceita:
      - get_orbital_elements(X, mu)        # X = [rx,ry,rz,vx,vy,vz]
      - get_orbital_elements(r, v, mu)

    Retorna sempre os elementos clássicos. Tratamento dos casos deg:
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
    e   = get_eccentricity(r, v, mu)  # com snap-to-zero leve
    inc = get_inclination(r, v, mu)

    # componentes circulares e bases do plano orbital
    e_raw, ex, ey, alpha_v_deg, p_hat, q_hat, hhat, nnorm = _circular_components(r, v, mu)

    # RAAN Ω (0 se equatorial)
    if nnorm > 0.0:
        n = np.cross(Bases.k, np.cross(r, v))
        Omega_deg = _wrap360_deg(_atan2(n[1], n[0]))
    else:
        Omega_deg = 0.0

    # flags de regime
    i_rad = np.deg2rad(inc)
    circular   = (e <= _EPS_E)
    equatorial = (i_rad <= _EPS_I) or (abs(inc - 180.0) <= np.rad2deg(_EPS_I))

    if not circular and not equatorial:
        # elíptica & inclinada: ω = atan2(ey, ex), ν = αᵥ − ω
        omega_deg = _wrap360_deg(np.arctan2(ey, ex))
        nu_deg    = (alpha_v_deg - omega_deg) % 360.0
        return OrbitalElements(
            major_axis=a,
            eccentricity=e,
            inclination=inc,
            ascending_node=Omega_deg,
            argument_of_perigee=omega_deg,
            true_anomaly=nu_deg,
            argument_of_latitude=alpha_v_deg   # <<< CORRETO: u = α_v
        )


    if circular and not equatorial:
        return OrbitalElements(
            major_axis=a,
            eccentricity=0.0,
            inclination=inc,
            ascending_node=Omega_deg,
            argument_of_perigee=0.0,
            true_anomaly=alpha_v_deg,
            argument_of_latitude=alpha_v_deg   # <<< inclui u
        )

    if (not circular) and equatorial:
        # elíptica-equatorial: ϖ=atan2(ey,ex) e Ω indefinido
        varpi_deg = _wrap360_deg(np.arctan2(ey, ex))
        # ν ainda pode ser definido a partir de e⃗
        e_vec = get_eccentricity_vector(r, v, mu)
        nu_deg = _wrap360_deg(_atan2(np.dot(np.cross(e_vec, r), hhat), np.dot(e_vec, r)))
        return OrbitalElements(
            major_axis=a, eccentricity=e, inclination=0.0 if inc < 90.0 else 180.0,
            ascending_node=0.0, argument_of_perigee=varpi_deg, true_anomaly=nu_deg
        )

    # circular-equatorial: λ; Ω=ω=0
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
    return 0.0 if e < _EPS_E else e  # snap-to-zero suave p/ casos quase-circulares

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
    """
    ω estável:
      - se e > EPS e não-equatorial: ω = atan2(ey, ex)
      - caso contrário: 0 (indefinido) — use ϖ fora se precisar
    """
    e, ex, ey, _, _, _, _, nnorm = _circular_components(r, v, mu)
    if e <= _EPS_E or nnorm == 0.0:
        return 0.0
    return _wrap360_deg(np.arctan2(ey, ex))

def get_true_anomaly(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    """
    Posição angular apropriada:
      - e>EPS e não-equatorial: ν = αᵥ − ω
      - circular-inclinada: u  (argumento de latitude)
      - circular-equatorial: λ  (longitude verdadeira)
    """
    e, ex, ey, alpha_v_deg, _, _, _, nnorm = _circular_components(r, v, mu)

    if e > _EPS_E and nnorm > 0.0:
        omega_deg = _wrap360_deg(np.arctan2(ey, ex))
        return (alpha_v_deg - omega_deg) % 360.0

    if nnorm > 0.0:
        # circular-inclinada: use u = αᵥ
        return alpha_v_deg

    # circular-equatorial → λ
    return _wrap360_deg(_atan2(r[1], r[0]))

def get_period(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    a = get_major_axis(r, v, mu)
    return 2.0*np.pi*np.sqrt(abs(a**3)/mu) if np.isfinite(a) else np.inf

# ================= elementos alternativos (opcionais) =================
def get_argument_of_latitude(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v)
    n = np.cross(Bases.k, h)

    dot_rn = np.dot(r, n)
    cross_rn = np.cross(r, n)

    sigma_rad = np.atan2(np.linalg.norm(cross_rn), dot_rn)

    sigma_deg = np.rad2deg(sigma_rad)

    return sigma_deg

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

# manter compatibilidade com grafia antiga
get_true_anormaly = get_true_anomaly
