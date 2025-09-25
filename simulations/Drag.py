# Drag.py
# Perturbação de arrasto atmosférico — unidades internas SI; saída em km/s^2.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# ----------------- Constantes da Terra -----------------
OMEGA_EARTH = 7.2921150e-5  # rad/s
R_E_KM      = 6378.1363     # km
R_E_M       = R_E_KM * 1000.0

@dataclass(frozen=True)
class DragParams:
    Cd: float = 2.2               # coeficiente de arrasto efetivo
    A_ref_m2: float = 0.02        # área de referência [m^2]
    use_atmo_rotation: bool = True
    # Modelo exponencial: rho ~ rho0 * exp(-(h - h0)/H)
    rho0_kg_m3: float = 3.614e-11 # ~ 200 km típico (ajuste conforme seu caso)
    h0_km: float = 200.0          # altitude de referência [km]
    H_km: float = 50.0            # altura de escala [km]

def _rho_exponential(h_km: float, p: DragParams) -> float:
    """Densidade exponencial simples [kg/m^3]."""
    return p.rho0_kg_m3 * np.exp(-(h_km - p.h0_km) / (p.H_km + 1e-32))

def _v_atmo_kmps(r_km: np.ndarray, p: DragParams) -> np.ndarray:
    """Velocidade da atmosfera (rotação) em km/s no frame inercial z||eixo Terra."""
    if not p.use_atmo_rotation:
        return np.zeros(3)
    omega = np.array([0.0, 0.0, OMEGA_EARTH])  # rad/s
    # (omega × r) → r em km → resultado em km/s
    return np.cross(omega, r_km)

def accel_drag(r_km: np.ndarray,
               v_kmps: np.ndarray,
               m_kg: float,
               params: DragParams = DragParams()) -> np.ndarray:
    """
    Aceleração de arrasto em km/s^2.
    Passo-a-passo:
      - converte r, v para SI
      - v_rel = v - v_atm
      - a_D = -1/2 * rho * Cd * A / m * |v_rel| * v_rel
      - devolve em km/s^2
    """
    r_km  = np.asarray(r_km,  dtype=float)
    v_kmps = np.asarray(v_kmps, dtype=float)
    m_kg  = float(max(m_kg, 1e-18))

    r_m   = 1000.0 * r_km
    v_mps = 1000.0 * v_kmps

    r_norm_m = float(np.linalg.norm(r_m) + 1e-32)
    h_km     = r_norm_m/1000.0 - R_E_KM
    rho      = _rho_exponential(h_km, params)

    v_atm_kmps = _v_atmo_kmps(r_km, params)
    v_rel_mps  = 1000.0 * (v_kmps - v_atm_kmps)
    Vrel       = float(np.linalg.norm(v_rel_mps) + 1e-32)

    q_over_m   = 0.5 * rho * params.Cd * params.A_ref_m2 / m_kg  # [1/m]
    a_drag_mps2 = - q_over_m * Vrel * v_rel_mps                   # [m/s^2]
    return a_drag_mps2 / 1000.0                                   # [km/s^2]
