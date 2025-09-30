# utils/orbital_elements.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

_EPS_E = 1e-8       # tolerância p/ circularidade
_EPS_I = 1e-8       # tolerância p/ equatorial

@dataclass
class OrbitalElements:
    a: float                   # semi-eixo maior [km]
    e: float                   # excentricidade
    i: float                   # inclinação [rad]
    Omega: float               # RAAN [rad]
    omega: float               # argumento do perigeu [rad]
    nu: float                  # anomalia verdadeira [rad]
    # Alternativos sempre definidos:
    u: float                   # argumento da latitude [rad]         (válido se e ~ 0)
    l_true: float              # longitude verdadeira [rad]          (válido se i ~ 0)
    lon_peri: float            # longitude do perigeu Π = Ω + ω [rad]
    energy_spec: float         # energia específica [km^2/s^2]
    h_vec: np.ndarray          # vetor momento angular específico [km^2/s]
    e_vec: np.ndarray          # vetor excentricidade
    r: np.ndarray              # posição ECI [km]
    v: np.ndarray              # velocidade ECI [km/s]

def _safe_acos(x: float) -> float:
    return float(np.arccos(np.clip(x, -1.0, 1.0)))

def from_rv(r: np.ndarray, v: np.ndarray, mu: float) -> OrbitalElements:
    r = np.asarray(r, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)
    rnorm = np.linalg.norm(r)
    vnorm = np.linalg.norm(v)

    # vetores fundamentais
    h = np.cross(r, v)
    hnorm = np.linalg.norm(h)
    if hnorm < 1e-12:
        raise ValueError("Momento angular ~0; estado inválido.")
    khat = np.array([0.0, 0.0, 1.0])
    n = np.cross(khat, h)
    nnorm = np.linalg.norm(n)

    # excentricidade e energia
    e_vec = (1.0/mu) * (np.cross(v, h) - mu * r / rnorm)
    e = float(np.linalg.norm(e_vec))
    eps = 0.5*vnorm**2 - mu/rnorm    # energia específica
    if abs(1 + 2*eps/ (mu/rnorm)) < 1e-12:
        # só para evitar divisão 0 em degenerescências absurdas
        pass
    a = -mu/(2.0*eps)

    # inclinação
    i = _safe_acos(h[2]/hnorm)

    # RAAN
    if nnorm > 1e-14:
        Omega = np.arctan2(n[1], n[0])  # atan2(n_y, n_x)
    else:
        Omega = 0.0  # equatorial -> indefinido, fixe 0 para consistência

    # argumento do perigeu
    if e > _EPS_E and nnorm > 1e-14:
        cosw = np.dot(n, e_vec)/(nnorm*e)
        sinw = np.dot(np.cross(n, e_vec), h)/(nnorm*e*hnorm)
        omega = np.arctan2(sinw, np.clip(cosw, -1.0, 1.0))
    else:
        omega = 0.0  # circular OU equatorial -> indefinido

    # anomalia verdadeira
    if e > _EPS_E:
        cosnu = np.dot(e_vec, r)/(e*rnorm)
        sinnu = np.dot(np.cross(e_vec, r), h)/(e*rnorm*hnorm)
        nu = np.arctan2(sinnu, np.clip(cosnu, -1.0, 1.0))
    else:
        # circular: use u = arg(latitude) = angle(n, r) se nnorm>0
        if nnorm > 1e-14:
            cosu = np.dot(n, r)/(nnorm*rnorm)
            sinu = np.dot(np.cross(n, r), h)/(nnorm*rnorm*hnorm)
            u = np.arctan2(sinu, np.clip(cosu, -1.0, 1.0))
        else:
            # circular e equatorial: use longitude verdadeira
            u = 0.0
        nu = 0.0  # não usado para e≈0; mantido para compatibilidade

    # elementos alternativos (definidos sempre)
    # u: argumento da latitude
    if (e <= _EPS_E) and (nnorm > 1e-14):
        u = float(u)  # da ramificação acima
    else:
        u = (omega + nu) % (2.0*np.pi)

    # l_true: longitude verdadeira (i≈0)
    if i <= _EPS_I:
        l_true = float(np.arctan2(r[1], r[0]))  # atan2(y, x)
    else:
        l_true = (Omega + omega + nu) % (2.0*np.pi)

    lon_peri = (Omega + omega) % (2.0*np.pi)

    return OrbitalElements(
        a=a, e=e, i=i, Omega=Omega % (2.0*np.pi), omega=omega % (2.0*np.pi), nu=nu % (2.0*np.pi),
        u=u % (2.0*np.pi), l_true=l_true % (2.0*np.pi), lon_peri=lon_peri,
        energy_spec=eps, h_vec=h, e_vec=e_vec, r=r, v=v
    )
