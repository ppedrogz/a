from __future__ import annotations
import numpy as np
from dataclasses import dataclass

Array = np.ndarray

@dataclass(frozen=True)
class EarthParams:
    """Parâmetros da Terra em km–s."""
    mu: float = 3.986004418e5   # km^3/s^2
    Re: float = 6378.1363       # km
    J2: float = 1.08262668e-3   # -

@dataclass(frozen=True)
class PerturbationFlags:
    """Liga/desliga de cada termo de perturbação (apenas J2 ativo por enquanto)."""
    j2: bool = True
    # futuros:
    # drag: bool = False


# ---------------------------
# Perturbação por achatamento
# ---------------------------
def accel_j2(r_vec: Array, mu: float, Re: float, J2: float) -> Array:
    """
    Aceleração perturbadora J2 (achatamento), em frame inercial com eixo z
    alinhado ao eixo de rotação da Terra.

    Parâmetros
    ----------
    r_vec : (3,) km    - vetor posição inercial
    mu    : km^3/s^2   - parâmetro gravitacional
    Re    : km         - raio equatorial
    J2    : -          - coeficiente zonal

    Retorno
    -------
    (3,) km/s^2
    """
    x, y, z = float(r_vec[0]), float(r_vec[1]), float(r_vec[2])
    r2 = x*x + y*y + z*z
    r  = (r2 + 0.0)**0.5 + 1e-32  # proteção numérica
    z2 = z*z

    fac = (3.0 * mu * J2 * (Re**2)) / (2.0 * r**5)
    cxy = (5.0 * z2 / r2) - 1.0
    cz  = (5.0 * z2 / r2) - 3.0

    return np.array([fac * x * cxy,
                     fac * y * cxy,
                     fac * z * cz], dtype=float)

# ----------------------------------------
# HUB de perturbações (só J2 por enquanto)
# ----------------------------------------
def accel_perturbations(r_vec: Array,
                        v_vec: Array | None = None,
                        t: float | None = None,
                        params: EarthParams = EarthParams(),
                        flags: PerturbationFlags = PerturbationFlags()) -> Array:
    """
    Soma das perturbações modeladas. Assinatura já preparada para futuros termos
    que dependem de v e/ou t (arrasto, SRP, J22, terceiros corpos).

    Retorna: (3,) km/s^2
    """
    a = np.zeros(3, dtype=float)

    if flags.j2:
        a += accel_j2(r_vec, params.mu, params.Re, params.J2)


    return a

# ----------------------------------------
# (Opcional) Kepler + J2 num só call
# ----------------------------------------
def accel_twobody(r_vec: Array, mu: float) -> Array:
    r = np.linalg.norm(r_vec) + 1e-32
    return -mu * r_vec / (r**3)

def accel_total_gravity_with_j2(r_vec: Array,
                                params: EarthParams = EarthParams(),
                                flags: PerturbationFlags = PerturbationFlags()) -> Array:
    """Gravidade central + perturbações (no momento, J2)."""
    return accel_twobody(r_vec, params.mu) + accel_perturbations(r_vec, None, None, params, flags)

__all__ = [
    "EarthParams",
    "PerturbationFlags",
    "accel_j2",
    "accel_perturbations",
    "accel_twobody",
    "accel_total_gravity_with_j2",
]