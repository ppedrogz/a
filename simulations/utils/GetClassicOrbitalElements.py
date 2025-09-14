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
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    eps = (v_norm**2)/2 - (mu/r_norm)
    a = -mu/(2*eps)

    return a


def get_eccentricity(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e = get_eccentricity_vector(r, v, mu)
    e_norm = np.linalg.norm(e)

    return e_norm


def get_inclination(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v)
    i_rad = np.acos(np.dot(Bases.k, h)/np.linalg.norm(h)) # dot of k_hat.h is the same as h[2] (z component of h)
    i = np.rad2deg(i_rad)

    return i


def get_ascending_node(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    h = np.cross(r, v)
    n = np.cross(Bases.k, h)
    Omega_rad = np.acos(np.dot(Bases.i, n)/np.linalg.norm(n))
    Omega = np.rad2deg(Omega_rad)

    return Omega


def get_argument_of_perigee(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e = get_eccentricity_vector(r, v, mu)
    h = np.cross(r, v)
    n = np.cross(Bases.k, h)

    omega_rad = np.acos(np.dot(n, e)/(n*np.linalg.norm(e)))
    omega = np.rad2deg(omega_rad)

    return omega


def get_true_anormaly(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> float:
    e = get_eccentricity_vector(r, v, mu)

    v_rad = np.acos(np.dot(e, r)/ (np.linalg.norm(e)*np.linalg.norm(r)))
    v = np.rad2deg(v_rad)

    return v


def get_eccentricity_vector(r: np.typing.NDArray, v: np.typing.NDArray, mu: float) -> np.typing.NDArray:
    h = np.cross(r, v)  # angular mommentum
    e = (np.cross(v, h)/mu) - (r/np.linalg.norm(r))

    return e