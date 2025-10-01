# simulations/achatamento.py
# Acelerações J2 e J22 (cartesiano, ECI z||spin); unidades: km, s, km^3/s^2 → km/s^2
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class EarthShapeParams:
    mu: float  = 3.986004418e5   # km^3/s^2
    Re: float  = 6378.1363       # km
    J2: float  = 1.08262668e-3   # -
    J22: float = 1.76600e-6      # módulo do grau-2 ordem-2 (compatível com ar_prs)

def _safe_norm(r: np.ndarray) -> float:
    n = float(np.linalg.norm(r))
    return n if n > 1e-32 else 1e-32

# ---------------- J2 ----------------
def accel_j2(r_km: np.ndarray, p: EarthShapeParams) -> np.ndarray:
    x, y, z = map(float, r_km)
    r = _safe_norm(r_km)
    r2 = r*r
    fac = 1.5 * p.J2 * (p.Re*p.Re) / (r2)   # = 1.5*J2*(Re^2/r^2)
    z2_over_r2 = (z*z) / r2
    cxy = fac * (5.0*z2_over_r2 - 1.0)
    cz  = fac * (5.0*z2_over_r2 - 3.0)
    base = p.mu / (r**3)
    return np.array([base * cxy * x, base * cxy * y, base * cz * z], dtype=float)

# ---------------- J22 (tesseral) ----------------
def accel_j22(r_km: np.ndarray, p: EarthShapeParams,
              lambdat_rad: float, lambda22_rad: float) -> np.ndarray:
    """
    Implementação via U22 = 3 mu Re^2 / r^5 * F, F = C22(x^2 - y^2) + 2 S22 x y
    com C22 = J22 cos(2(λt-λ22)), S22 = J22 sin(2(λt-λ22)).
    """
    x, y, z = map(float, r_km)
    r = _safe_norm(r_km)
    r2, r5, r7 = r*r, r**5, r**7

    # Coeficientes efetivos no tempo
    psi  = 2.0 * (lambdat_rad - lambda22_rad)
    C22  = p.J22 * np.cos(psi)
    S22  = p.J22 * np.sin(psi)

    # F e derivadas
    F    = C22 * (x*x - y*y) + 2.0 * S22 * x * y

    # -∇U: usar as expressões fechadas (estáveis em e→0)
    common = 3.0 * p.mu * (p.Re**2)
    ax = common * ( (5.0 * x * F) / r7 - (2.0*C22*x + 2.0*S22*y) / r5 )
    ay = common * ( (5.0 * y * F) / r7 - (-2.0*C22*y + 2.0*S22*x) / r5 )
    az = common * ( (5.0 * z * F) / r7 )

    return np.array([ax, ay, az], dtype=float)

def accel_achatamento_total(r_km: np.ndarray, p: EarthShapeParams,
                            lambdat_rad: float = 0.0, lambda22_rad: float = 0.0,
                            use_j2: bool = True, use_j22: bool = False) -> np.ndarray:
    a = np.zeros(3, dtype=float)
    if use_j2:
        a += accel_j2(r_km, p)
    if use_j22:
        a += accel_j22(r_km, p, lambdat_rad, lambda22_rad)
    return a

__all__ = ["EarthShapeParams", "accel_achatamento_total", "accel_j2", "accel_j22"]
