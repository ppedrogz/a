import numpy as np
from dataclasses import dataclass
from utils.constants import Bases


@dataclass
class OrbitalElements:
    major_axis: float
    eccentricity: float
    inclination: float
    ascending_node: float  #  Right ascension of the ascending node
    argument_of_perigee: float
    true_anomaly: float


def get_orbital_elements(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> OrbitalElements:
    orbital_elements = OrbitalElements(
        major_axis = get_major_axis(r, v, mu),
        eccentricity = get_eccentricity(r, v, mu),
        inclination = get_inclination(r, v, mu),
        ascending_node = get_ascending_node(r, v, mu),
        argument_of_perigee = get_argument_of_perigee(r, v, mu),
        true_anomaly = get_true_anormaly(r, v, mu)
    )

    return orbital_elements


def get_major_axis(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    # energia específica: ε = v^2/2 - μ/r  (kepleriano)
    r_norm = np.linalg.norm(r)
    v2 = float(np.dot(v, v))
    eps = 0.5*v2 - (mu / r_norm)
    # a = - μ / (2 ε)   (para ε<0; se ε≈0, parabólico → trate separadamente se quiser)
    return -mu / (2.0*eps)

def get_eccentricity(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    # forma invariante: e = sqrt(1 + (2 ε h^2)/μ^2)
    h_vec = np.cross(r, v)
    h2 = float(np.dot(h_vec, h_vec))
    r_norm = np.linalg.norm(r)
    v2 = float(np.dot(v, v))
    eps = 0.5*v2 - (mu / r_norm)
    val = 1.0 + (2.0*eps*h2)/(mu*mu)
    # proteção numérica contra ruído (evita e>1+1e-12 por arredondamento)
    return float(np.sqrt(max(val, 0.0)))



def get_inclination(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v)
    i_rad = np.acos(np.dot(Bases.k, h)/np.linalg.norm(h)) # dot of k_hat.h is the same as h[2] (z component of h)
    i = np.rad2deg(i_rad)

    return i


def get_ascending_node(r, v, mu):
    h = np.cross(r, v)
    n = np.cross(Bases.k, h)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return 0.0   # equatorial → indefinido
    return float(np.degrees(np.arctan2(n[1], n[0])) % 360.0)

def get_argument_of_perigee(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e_vec = get_eccentricity_vector(r, v, mu)
    e = np.linalg.norm(e_vec)
    h = np.cross(r, v)
    n = np.cross(Bases.k, h)

    if np.linalg.norm(n) > 1e-10:
        omega_rad = np.arccos(np.dot(n, e_vec)/(np.linalg.norm(n)*e))
        if e_vec[2] < 0:
            omega_rad = 2*np.pi - omega_rad
    else:
        omega_rad = 0

    omega = np.rad2deg(omega_rad)

    return omega


def get_true_anormaly(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e = get_eccentricity_vector(r, v, mu)

    denom = np.linalg.norm(e) * np.linalg.norm(r)
    cos_v = np.dot(e, r) / denom
    cos_v = np.clip(cos_v, -1.0, 1.0)

    v_rad = np.arccos(cos_v)  # [0, π]

    if np.dot(r, v) < 0.0:
        v_rad = 2.0 * np.pi - v_rad

    v = np.degrees(v_rad)  # converte para graus
    return v


def get_eccentricity_vector(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> np.typing.NDArray:
    h = np.cross(r, v)  # angular mommentum
    e = (np.cross(v, h)/mu) - (r/np.linalg.norm(r))

    return e

def get_period(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    a = get_major_axis(r, v, mu)
    period = 2*np.pi*np.sqrt((a**3)/mu)

    return period


# KEPLERIAN ELEMENTS
def get_eccentric_anomaly(theta, e) -> float:
    E_sin = np.sqrt(1-e**2)*np.sin(theta)
    E_cos = 1 + e*np.cos(theta)

    E = np.atan2(E_sin, E_cos)

    return E
