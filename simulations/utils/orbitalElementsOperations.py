# utils/orbitalElementsOperations.py
from __future__ import annotations
import numpy as np
from utils.orbital_elements import from_rv, OrbitalElements
from utils.angles import RAD2DEG

def get_orbital_elements(r: np.ndarray, v: np.ndarray, mu: float) -> OrbitalElements:
    return from_rv(r, v, mu)

# Funções auxiliares já no formato "float em unidades usuais" p/ plot
def get_major_axis(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    return from_rv(r, v, mu).a

def get_eccentricity(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    return from_rv(r, v, mu).e

def get_inclination(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    return RAD2DEG * from_rv(r, v, mu).i

def get_ascending_node(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    return RAD2DEG * from_rv(r, v, mu).Omega

def get_argument_of_perigee(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    return RAD2DEG * from_rv(r, v, mu).omega

def get_true_anomaly(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    return RAD2DEG * from_rv(r, v, mu).nu

def get_argument_of_latitude(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    return RAD2DEG * from_rv(r, v, mu).u

def get_true_longitude(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    return RAD2DEG * from_rv(r, v, mu).l_true

def get_longitude_of_periapsis(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    return RAD2DEG * from_rv(r, v, mu).lon_peri

def get_specific_energy(r: np.ndarray, v: np.ndarray, mu: float) -> float:
    return from_rv(r, v, mu).energy_spec
