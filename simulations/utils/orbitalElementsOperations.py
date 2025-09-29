import numpy as np
from utils.types import OrbitalElements
from .constants import Bases

# ================= utilidades numéricas seguras =================
def _safe_norm(x, eps: float = 1e-18) -> float:
    """Norma segura: devolve 0.0 se ||x|| <= eps (evita divisão por zero)."""
    n = float(np.linalg.norm(x))
    return n if n > eps else 0.0

def _safe_unit(x, eps: float = 1e-18):
    """Versor seguro: devolve zeros se o vetor tiver norma ~0."""
    n = _safe_norm(x, eps)
    return x/n if n > 0.0 else np.zeros_like(x)

def _clamp(x, a=-1.0, b=1.0):
    return a if x < a else (b if x > b else x)

def _atan2(y: float, x: float) -> float:
    return float(np.arctan2(y, x))

# ===== Robustez numérica para casos quase-circulares/equatoriais =====
_EPS_E = 2e-3          # limiar de circularidade (e <= 0.002 -> tratar como circular)
_EPS_I_DEG = 0.05      # limiar de equatorialidade em graus
_EPS_I = np.deg2rad(_EPS_I_DEG)
_TWOPI = 2.0*np.pi

def _wrap_0_2pi(theta: float) -> float:
    """Em radianos -> [0, 2π)."""
    return float(np.mod(theta, _TWOPI))

def _wrap360_deg(theta_rad: float) -> float:
    """Recebe ângulo em radianos e devolve em graus no intervalo [0, 360)."""
    return float(np.degrees(np.mod(theta_rad, _TWOPI)))

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
      αᵥ = ν + ω = atan2(r·q̂, r·p̂)  (retornado em GRAUS).
    """
    e_vec = get_eccentricity_vector(r, v, mu)
    e = _safe_norm(e_vec)
    p_hat, q_hat, hhat, nnorm = _orbital_frame(r, v)
    ex = float(np.dot(e_vec, p_hat))
    ey = float(np.dot(e_vec, q_hat))
    alpha_v = _atan2(np.dot(r, q_hat), np.dot(r, p_hat))   # rad
    alpha_v_deg = _wrap360_deg(alpha_v)                    # -> graus [0,360)
    return e, ex, ey, alpha_v_deg, p_hat, q_hat, hhat, nnorm

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

    # RAAN Ω (0 se equatorial) — em graus
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
            major_axis=a, eccentricity=e, inclination=inc,
            ascending_node=Omega_deg, argument_of_perigee=omega_deg, true_anomaly=nu_deg
        )

    if circular and not equatorial:
        # circular-inclinada: u no lugar de ν; ω indefinido
        return OrbitalElements(
            major_axis=a, eccentricity=0.0, inclination=inc,
            ascending_node=Omega_deg, argument_of_perigee=0.0, true_anomaly=alpha_v_deg
        )

    if (not circular) and equatorial:
        # elíptica-equatorial: ϖ=atan2(ey,ex) e Ω indefinido
        varpi_deg = _wrap360_deg(np.arctan2(ey, ex))
        # ν ainda pode ser definido a partir de e⃗ e ĥ (ângulo entre e⃗ e r no plano orbital)
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
    eps = 0.5*(v_norm**2) - (mu/max(r_norm, 1e-32))
    return -mu/(2.0*eps) if eps != 0.0 else np.inf

def get_eccentricity_vector(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> np.typing.NDArray:
    h = np.cross(r, v)
    return (np.cross(v, h)/mu) - (r/(max(_safe_norm(r), 1e-32)))

def get_eccentricity(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e = _safe_norm(get_eccentricity_vector(r, v, mu))
    return 0.0 if e < _EPS_E else e  # snap-to-zero suave p/ casos quase-circulares

def get_inclination(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v)
    h_norm = _safe_norm(h)
    cos_i = h[2]/h_norm if h_norm > 0.0 else 1.0
    i_rad = np.arccos(_clamp(cos_i))
    i_deg = float(np.degrees(i_rad))
    if i_deg < 1e-6: return 0.0
    if abs(i_deg - 180.0) < 1e-6: return 180.0
    return i_deg

def get_ascending_node(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    """RAAN Ω (robusto com atan2). Se equatorial (|i| ~ 0), devolve 0 por convenção.
       Retorna **em radianos** no intervalo [0,2π) (função utilitária antiga; prefira Ω em graus no pipeline principal)."""
    h = np.cross(r, v)
    n = np.cross(np.array([0.0, 0.0, 1.0]), h)
    nx, ny = n[0], n[1]
    n_norm = _safe_norm(n)

    # Inclinação
    i = np.arccos(np.clip(h[2] / max(_safe_norm(h), 1e-15), -1.0, 1.0))

    if i < _EPS_I or abs(i - np.pi) < _EPS_I:
        # Equatorial: Ω indefinido -> adote 0° (aqui retornamos 0 rad)
        return 0.0

    Om = np.arctan2(ny, nx)
    return _wrap_0_2pi(Om)

def get_argument_of_perigee(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    """ω (robusto). Em órbita circular, ω é indefinido -> retorna 0 (u ficará em ν). Retorna em rad no intervalo [0,2π)."""
    h = np.cross(r, v)
    n = np.cross(np.array([0.0, 0.0, 1.0]), h)
    n_norm = _safe_norm(n)

    e_vec = (np.cross(v, h) / mu) - (r / max(_safe_norm(r), 1e-15))
    e = _safe_norm(e_vec)

    # Base no plano orbital
    h_hat = h / max(_safe_norm(h), 1e-15)
    if n_norm > 0.0:
        p_hat = n / n_norm
    else:
        # Equatorial: defina p_hat arbitrário no plano XY
        p_hat = np.array([1.0, 0.0, 0.0])
    q_hat = np.cross(h_hat, p_hat)

    if e <= _EPS_E:
        # Circular: ω indefinido -> 0 por convenção
        return 0.0

    # ω = atan2(e·q_hat, e·p_hat)
    w = np.arctan2(np.dot(e_vec, q_hat), np.dot(e_vec, p_hat))
    return _wrap_0_2pi(w)

def get_true_anormaly(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    """
    ν (robusto) em GRAUS [0,360). Para e ~ 0 usa u = atan2(r·q̂, r·p̂).
    Para e > 0 usa ν = atan2(r·ê_q, r·ê_p) - ω (em base p̂,q̂).
    """
    h = np.cross(r, v)
    n = np.cross(np.array([0.0, 0.0, 1.0]), h)
    n_norm = _safe_norm(n)

    e_vec = (np.cross(v, h) / mu) - (r / max(_safe_norm(r), 1e-15))
    e = _safe_norm(e_vec)

    h_hat = h / max(_safe_norm(h), 1e-15)
    if n_norm > 0.0:
        p_hat = n / n_norm
    else:
        p_hat = np.array([1.0, 0.0, 0.0])
    q_hat = np.cross(h_hat, p_hat)

    rp = float(np.dot(r, p_hat))
    rq = float(np.dot(r, q_hat))

    if e <= _EPS_E:
        # Circular: argumento de latitude u
        u = np.arctan2(rq, rp)                       # rad
        return _wrap360_deg(u)                       # -> graus [0,360)

    # Elíptica
    w = get_argument_of_perigee(r, v, mu)            # rad
    nu_geo = np.arctan2(rq, rp) - w                  # rad
    return _wrap360_deg(nu_geo)                      # -> graus [0,360)

# Alias ortográfico útil (opcional):
get_true_anomaly = get_true_anormaly

def get_period(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    a = get_major_axis(r, v, mu)
    return float(2.0*np.pi*np.sqrt(abs(a**3)/mu)) if np.isfinite(a) else np.inf

# ================= elementos alternativos (opcionais) =================
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
