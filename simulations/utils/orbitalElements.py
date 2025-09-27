import numpy as np
from utils.types import OrbitalElements
from .constants import Bases


def get_orbital_elements(X: np.typing.NDArray, mu: float) -> OrbitalElements:
    r = X[0:3]
    v = X[3:6]

    orbital_elements = OrbitalElements(
        major_axis=get_major_axis(r, v, mu),
        eccentricity=get_eccentricity(r, v, mu),
        inclination=get_inclination(r, v, mu),
        ascending_node=get_ascending_node(r, v, mu),
        argument_of_perigee=get_argument_of_perigee(r, v, mu),
        true_anormaly=get_true_anomaly(r, v, mu)
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
    i_rad = np.arccos(np.dot(Bases.k, h)/np.linalg.norm(h))  # dot of k_hat.h is the same as h[2] (z component of h)
    i = np.rad2deg(i_rad)

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

    dot_er = np.dot(e, r)
    cross_er = np.cross(e, r)
    theta_rad = np.atan2(np.linalg.norm(cross_er), dot_er)
    theta = np.rad2deg(theta_rad)

    return theta


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
