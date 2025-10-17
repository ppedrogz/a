# Drag.py
# Perturbação de arrasto atmosférico — unidades internas SI; saída em km/s^2.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

# ----------------- Constantes da Terra -----------------
OMEGA_EARTH = 7.2921150e-5  # rad/s
R_E_KM      = 6378.1363     # km
R_E_M       = R_E_KM * 1000.0

# ----------------- Tabela "TD-88 (aprox.)" -----------------
# Densidades médias (kg/m^3) por altitude (km): 100–1000 km.
# Interpolação log-linear entre nós. Fora do intervalo: extrapolação exponencial.
_H_KM_GRID = np.array([100, 125, 150, 175, 200, 225, 250, 275,
                       300, 325, 350, 375, 400, 450, 500, 600,
                       700, 800, 900, 1000], dtype=float)

_RHO_KGM3_GRID = np.array([
    5.606e-07, 1.916e-07, 6.421e-09, 1.794e-09, 5.297e-10,
    1.950e-10, 7.248e-11, 2.969e-11, 1.170e-11, 4.932e-12,
    2.390e-12, 1.191e-12, 6.086e-13, 1.921e-13, 7.014e-14,
    1.454e-14, 3.614e-15, 1.170e-15, 4.569e-16, 1.723e-16
], dtype=float)

def _rho_td88_like(h_km: float) -> float:
    """Densidade 'TD-88 (aprox.)' por interpolação log-linear (100–1000 km)."""
    if h_km <= _H_KM_GRID[0]:
        # extrapolação exponencial abaixo de 100 km (clamp suave)
        H = 7.0  # escala efetiva ~troposfera/estratosfera (valor típico)
        rho0 = _RHO_KGM3_GRID[0]
        return float(rho0 * np.exp(-( _H_KM_GRID[0] - h_km ) / max(H, 1e-6)))
    if h_km >= _H_KM_GRID[-1]:
        # extrapolação exponencial acima de 1000 km
        H = 200.0  # escala efetiva alta
        rhoN = _RHO_KGM3_GRID[-1]
        return float(rhoN * np.exp(-( h_km - _H_KM_GRID[-1] ) / max(H, 1e-6)))

    # Interpolação log-linear
    i = np.searchsorted(_H_KM_GRID, h_km) - 1
    i = int(np.clip(i, 0, len(_H_KM_GRID) - 2))
    h0, h1 = _H_KM_GRID[i], _H_KM_GRID[i+1]
    r0, r1 = _RHO_KGM3_GRID[i], _RHO_KGM3_GRID[i+1]
    t = (h_km - h0) / (h1 - h0 + 1e-32)
    log_r = np.log(r0 + 1e-300) * (1 - t) + np.log(r1 + 1e-300) * t
    return float(np.exp(log_r))

@dataclass(frozen=True)
class DragParams:
    # Aerodinâmica
    Cd: float = 2.2
    # 12U (face quadrada nominal 226.3 mm): A_ref ≈ (0.2263 m)^2 ≈ 0.0512 m²
    # Fontes de referência (dimensões típicas CDS): EnduroSat/C3S/EXOpod.
    A_ref_m2: float = 0.2263 * 0.2263
    # Atmosfera co-rotante?
    use_atmo_rotation: bool = True
    # Modelo atmosférico: 'td88' (aprox.) ou 'exponential'
    model: Literal['td88', 'exponential'] = 'td88'

    # Parâmetros do exponencial simples (mantidos para compatibilidade)
    rho0_kg_m3: float = 3.614e-11
    h0_km: float = 200.0
    H_km: float = 50.0

    # Segurança numérica
    rho_min: float = 0.0
    rho_max: float = 1.5          # clamp superior (~ ar ao nível do mar)
    h_cut_km: float = 1200.0      # acima disso, zera densidade

    # (Opcional) razão área/massa fixa; se None, usa A_ref_m2 / m_kg
    gamma_m2_per_kg: Optional[float] = None

def _rho_exponential(h_km: float, p: DragParams) -> float:
    if h_km >= p.h_cut_km:
        return 0.0
    h_eff = max(h_km, 0.0)
    rho = p.rho0_kg_m3 * np.exp(-(h_eff - p.h0_km) / (p.H_km + 1e-32))
    return float(np.clip(rho, p.rho_min, p.rho_max))

def _rho_model(h_km: float, p: DragParams) -> float:
    if h_km >= p.h_cut_km:
        return 0.0
    if p.model == 'td88':
        return float(np.clip(_rho_td88_like(h_km), p.rho_min, p.rho_max))
    else:
        return _rho_exponential(h_km, p)

def _v_atmo_kmps(r_km: np.ndarray, p: DragParams) -> np.ndarray:
    """Velocidade da atmosfera (rotação) em km/s no frame inercial z||eixo Terra."""
    if not p.use_atmo_rotation:
        return np.zeros(3)
    omega = np.array([0.0, 0.0, OMEGA_EARTH])  # rad/s
    # (omega × r) → r em km → resultado em km/s
    return np.cross(omega, r_km)

def accel_drag(r_km: np.ndarray,
               v_kmps: np.ndarray,
               m_kg: float = 20.0,
               params: DragParams = DragParams()) -> np.ndarray:
    """
    Aceleração de arrasto (km/s^2).
    - r_km, v_kmps: posição/velocidade ECI (km, km/s).
    - m_kg: massa do satélite [padrão: 20 kg].
    - params: ver DragParams (modelo 'td88' por padrão, A_ref 12U).
    """
    r_km   = np.asarray(r_km,  dtype=float)
    v_kmps = np.asarray(v_kmps, dtype=float)
    if not np.all(np.isfinite(r_km)) or not np.all(np.isfinite(v_kmps)):
        return np.zeros(3)

    m_kg = float(max(m_kg, 1e-12))

    r_m   = 1000.0 * r_km
    v_mps = 1000.0 * v_kmps

    r_norm_m = float(np.linalg.norm(r_m) + 1e-32)
    h_km     = r_norm_m/1000.0 - R_E_KM
    rho      = _rho_model(h_km, params)

    v_atm_kmps = _v_atmo_kmps(r_km, params)     # km/s
    v_rel_mps  = 1000.0 * (v_kmps - v_atm_kmps) # m/s
    Vrel       = float(np.linalg.norm(v_rel_mps) + 1e-32)

    # Permite usar γ = A/m diretamente, se fornecido
    if params.gamma_m2_per_kg is not None:
        gamma = max(params.gamma_m2_per_kg, 0.0)
    else:
        gamma = max(params.A_ref_m2, 0.0) / m_kg

    a_drag_mps2 = -0.5 * rho * params.Cd * gamma * Vrel * v_rel_mps  # m/s^2
    return a_drag_mps2 / 1000.0                                       # km/s^2
