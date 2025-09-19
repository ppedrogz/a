# main_constelacao.py  (exemplo robusto: chama perturbações só pelo main)
import numpy as np
import matplotlib.pyplot as plt

# seus módulos de simulação
import sat_vh_up as sat_vh_up
import sat_v_only as sat_v_only
import sat_vh_down as sat_vh_down

# ========================= roda as três simulações =========================
print("Simulando satélite VH UP...")
res_up = sat_vh_up.simulate()

print("Simulando satélite V ONLY...")
res_v = sat_v_only.simulate()

print("Simulando satélite VH DOWN...")
res_down = sat_vh_down.simulate()


t_up, X_up, nus_up, incs_up, elems_up = res_up
t_v, X_v, nus_v, incs_v, elems_v = res_v
t_down, X_down, nus_down, incs_down, elems_down = res_down
# ----------------- Gráficos comparativos -----------------
t_up, X_up, nus_up, incs_up, elems_up = sat_vh_up.simulate()
t_v, X_v, nus_v, incs_v, elems_v = sat_v_only.simulate()
t_down, X_down, nus_down, incs_down, elems_down = sat_vh_down.simulate()

from perturbations import EarthParams, diagnose_orbital_changes

P = EarthParams()  # (mu, Re, J2)

print("\n##### Diagnósticos de elementos (efeitos compatíveis com J2) #####")
diag_up   = diagnose_orbital_changes(t_up,   X_up,   params=P, label="VH UP")
diag_v    = diagnose_orbital_changes(t_v,    X_v,    params=P, label="V ONLY")
diag_down = diagnose_orbital_changes(t_down, X_down, params=P, label="VH DOWN")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
earth_radius = 6378.0

u, vgrid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = earth_radius * np.cos(u)*np.sin(vgrid)
y_e = earth_radius * np.sin(u)*np.sin(vgrid)
z_e = earth_radius * np.cos(vgrid)
ax.plot_wireframe(x_e, y_e, z_e, color="g")
ax.set_box_aspect([1, 1, 1])

ax.plot3D(X_up[0, :],   X_up[1, :],   X_up[2, :],   'b-', label="VH Up")
ax.plot3D(X_v[0, :],    X_v[1, :],    X_v[2, :],    'g-', label="V Only")
ax.plot3D(X_down[0, :], X_down[1, :], X_down[2, :], 'r-', label="VH Down")

ax.set_title("Constelação - 3 Satélites")
ax.legend()
plt.show()

# Inclinação vs anomalia verdadeira
plt.figure()
plt.plot(nus_up,   incs_up,   'b.', ms=0.7, label="VH Up")
plt.plot(nus_v,    incs_v,    'g.', ms=0.7, label="V Only")
plt.plot(nus_down, incs_down, 'r.', ms=0.7, label="VH Down")
plt.xlim(0, 360)
plt.xlabel("Anomalia Verdadeira ν [graus]")
plt.ylabel("Inclinação i [graus]")
plt.title("Inclinação vs ν - Comparação dos Satélites")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Massa vs tempo
plt.figure()
plt.plot(t_up,   X_up[6, :],   'b-', label="VH Up")
plt.plot(t_v,    X_v[6, :],    'g-', label="V Only")
plt.plot(t_down, X_down[6, :], 'r-', label="VH Down")
plt.xlabel("Tempo [s]")
plt.ylabel("Massa [kg]")
plt.title("Consumo de Propelente")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ==================== ADENDO: COMPARAÇÃO COM x SEM J2 (não remove nada acima) ====================
print("\n##### Adendo: Comparação COM x SEM J2 (gráficos de influência) #####")

from utils.GetClassicOrbitalElements import get_orbital_elements, get_true_anormaly, get_inclination

# --- parâmetros globais (mu, Re)
try:
    from perturbations import EarthParams
    _P = EarthParams()
    _MU = getattr(_P, "mu", 3.986e5)    # km^3/s^2
    _RE = getattr(_P, "Re", 6378.0)     # km
except Exception:
    _MU = 3.986e5
    _RE = 6378.0

def _call_sim_with_flag(module, flag: bool):
    """Tenta chamar simulate(use_perturbations=flag) ou simulate(j2=flag).
    Se não existir, retorna None (o caller usa fallback 2-corpos)."""
    try:
        return module.simulate(use_perturbations=flag)
    except TypeError:
        pass
    try:
        return module.simulate(j2=flag)
    except TypeError:
        pass
    return None

def _two_body_propagate(x0, t, mu=_MU):
    """Fallback SEM PERTURBAÇÃO: 2-corpos (sem J2/empuxo). Mantém massa constante se existir."""
    from scipy.integrate import solve_ivp
    n = len(x0)
    _has_mass = n >= 7
    def f(_t, x):
        r = x[0:3]; v = x[3:6]
        rn = np.linalg.norm(r) + 1e-32
        a = -(mu/rn**3)*r
        dx = np.zeros_like(x)
        dx[0:3] = v
        dx[3:6] = a
        if _has_mass:
            dx[6] = 0.0
        return dx
    sol = solve_ivp(f, (t[0], t[-1]), x0, t_eval=t, method='RK45')
    return sol.y

def _elements_series(t, X, mu=_MU):
    """Séries de elementos clássicos: ν, i, Ω, ω, e, a (em graus onde apropriado)."""
    N = X.shape[1]
    nus = np.empty(N); incs = np.empty(N)
    RAAN = np.empty(N); argp = np.empty(N)
    ecc = np.empty(N);  sma  = np.empty(N)
    for k in range(N):
        r = X[0:3, k]; v = X[3:6, k]
        el = get_orbital_elements(r, v, mu)
        nus[k]  = get_true_anormaly(r, v, mu)     # [0,360)
        incs[k] = el.inclination
        RAAN[k] = el.ascending_node
        argp[k] = el.argument_of_perigee
        ecc[k]  = el.eccentricity
        sma[k]  = el.major_axis
    return nus, incs, RAAN, argp, ecc, sma

def _unwrap_deg(a_deg):
    """Desenrola série angular em graus (evita saltos 360->0)."""
    return np.degrees(np.unwrap(np.radians(a_deg)))

def _plot_i_vs_nu_segmentado(nu_deg, inc_deg, *, ax=None, color="k", label=None, **kw):
    """Plota i(ν) em segmentos entre wraps para não ligar 360->0, mantendo cor única."""
    nu_deg = np.asarray(nu_deg, float); inc_deg = np.asarray(inc_deg, float)
    if ax is None:
        fig, ax = plt.subplots()
    dn = np.diff(nu_deg)
    wraps = np.where(dn < -180.0)[0]
    start = 0; first = True
    for w in wraps:
        ax.plot(nu_deg[start:w+1], inc_deg[start:w+1], color=color,
                label=(label if first else None), **kw)
        first = False
        start = w + 1
    ax.plot(nu_deg[start:], inc_deg[start:], color=color,
            label=(label if first else None), **kw)
    ax.set_xlim(0, 360)
    ax.set_xlabel("Anomalia Verdadeira ν [graus]")
    ax.set_ylabel("Inclinação i [graus]")
    ax.grid(alpha=0.3)
    return ax

# --------- séries COM perturbação (já calculadas acima): recalcula Ω, ω para gráficos
print("Computando séries COM perturbação para gráficos J2...")
Nu_up_on,  I_up_on,  Om_up_on,  W_up_on,  e_up_on,  a_up_on  = _elements_series(t_up,   X_up,   _MU)
Nu_v_on,   I_v_on,   Om_v_on,   W_v_on,   e_v_on,   a_v_on   = _elements_series(t_v,    X_v,    _MU)
Nu_dn_on,  I_dn_on,  Om_dn_on,  W_dn_on,  e_dn_on,  a_dn_on  = _elements_series(t_down, X_down, _MU)

# --------- constrói referências SEM perturbação (ideal: módulos aceitando flag; senão 2-corpos)
print("Gerando referências SEM perturbação (J2 OFF)...")
_res_up_off  = _call_sim_with_flag(sat_vh_up,   False)
_res_v_off   = _call_sim_with_flag(sat_v_only,  False)
_res_dn_off  = _call_sim_with_flag(sat_vh_down, False)

if _res_up_off is None:
    X_up_off  = _two_body_propagate(X_up[:, 0].copy(), t_up,   _MU)
    Nu_up_off, I_up_off, Om_up_off, W_up_off, e_up_off, a_up_off = _elements_series(t_up,   X_up_off,  _MU)
else:
    t_up_off, X_up_off, Nu_up_off, I_up_off, _ = _res_up_off
    _, _, Om_up_off, W_up_off, e_up_off, a_up_off = _elements_series(t_up_off, X_up_off, _MU)

if _res_v_off is None:
    X_v_off   = _two_body_propagate(X_v[:, 0].copy(),  t_v,    _MU)
    Nu_v_off,  I_v_off,  Om_v_off,  W_v_off,  e_v_off,  a_v_off  = _elements_series(t_v,    X_v_off,   _MU)
else:
    t_v_off, X_v_off, Nu_v_off, I_v_off, _ = _res_v_off
    _, _, Om_v_off, W_v_off, e_v_off, a_v_off = _elements_series(t_v_off, X_v_off, _MU)

if _res_dn_off is None:
    X_dn_off  = _two_body_propagate(X_down[:, 0].copy(), t_down, _MU)
    Nu_dn_off, I_dn_off, Om_dn_off, W_dn_off, e_dn_off, a_dn_off = _elements_series(t_down, X_dn_off,  _MU)
else:
    t_dn_off, X_dn_off, Nu_dn_off, I_dn_off, _ = _res_dn_off
    _, _, Om_dn_off, W_dn_off, e_dn_off, a_dn_off = _elements_series(t_dn_off, X_dn_off, _MU)

# --------- 3D: COM (linha cheia) vs SEM (tracejada) para cada satélite
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
u, vgrid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = _RE * np.cos(u)*np.sin(vgrid)
y_e = _RE * np.sin(u)*np.sin(vgrid)
z_e = _RE * np.cos(vgrid)
ax.plot_wireframe(x_e, y_e, z_e, color="g", alpha=0.3)
ax.set_box_aspect([1,1,1])

ax.plot3D(X_up[0,:],   X_up[1,:],   X_up[2,:],   '-',  color="#1f77b4", label="VH Up  (J2 ON)")
ax.plot3D(X_up_off[0,:],X_up_off[1,:],X_up_off[2,:],'--', color="#1f77b4", label="VH Up  (J2 OFF)")

ax.plot3D(X_v[0,:],    X_v[1,:],    X_v[2,:],    '-',  color="#2ca02c", label="V Only (J2 ON)")
ax.plot3D(X_v_off[0,:],X_v_off[1,:],X_v_off[2,:],'--', color="#2ca02c", label="V Only (J2 OFF)")

ax.plot3D(X_down[0,:], X_down[1,:], X_down[2,:], '-',  color="#d62728", label="VH Down (J2 ON)")
ax.plot3D(X_dn_off[0,:],X_dn_off[1,:],X_dn_off[2,:],'--', color="#d62728", label="VH Down (J2 OFF)")

ax.set_title("COM (linha cheia) vs SEM J2 (tracejada)")
ax.legend()
plt.show()

# --------- RAAN(t): deriva secular por J2 (desenrolado)
plt.figure()
plt.plot(t_up,   _unwrap_deg(Om_up_on),  '-',  color="#1f77b4", lw=1.2, label="VH Up  (J2 ON)")
plt.plot(t_up,   _unwrap_deg(Om_up_off), '--', color="#1f77b4", lw=1.2, label="VH Up  (J2 OFF)")
plt.plot(t_v,    _unwrap_deg(Om_v_on),   '-',  color="#2ca02c", lw=1.2, label="V Only (J2 ON)")
plt.plot(t_v,    _unwrap_deg(Om_v_off),  '--', color="#2ca02c", lw=1.2, label="V Only (J2 OFF)")
plt.plot(t_down, _unwrap_deg(Om_dn_on),  '-',  color="#d62728", lw=1.2, label="VH Down (J2 ON)")
plt.plot(t_down, _unwrap_deg(Om_dn_off), '--', color="#d62728", lw=1.2, label="VH Down (J2 OFF)")
plt.xlabel("Tempo [s]"); plt.ylabel("Ω (RAAN) [graus] (desenrolado)")
plt.title("Deriva secular de Ω (efeito J2)")
plt.grid(alpha=0.3); plt.legend(); plt.show()

# --------- Inclinação i(t): J2 não tem termo secular em i (diferenças são ondulações)
plt.figure()
plt.plot(t_up,   I_up_on,  '-',  color="#1f77b4", lw=1.2, label="VH Up  (J2 ON)")
plt.plot(t_up,   I_up_off, '--', color="#1f77b4", lw=1.2, label="VH Up  (J2 OFF)")
plt.plot(t_v,    I_v_on,   '-',  color="#2ca02c", lw=1.2, label="V Only (J2 ON)")
plt.plot(t_v,    I_v_off,  '--', color="#2ca02c", lw=1.2, label="V Only (J2 OFF)")
plt.plot(t_down, I_dn_on,  '-',  color="#d62728", lw=1.2, label="VH Down (J2 ON)")
plt.plot(t_down, I_dn_off, '--', color="#d62728", lw=1.2, label="VH Down (J2 OFF)")
plt.xlabel("Tempo [s]"); plt.ylabel("Inclinação i [graus]")
plt.title("Inclinação i(t): COM x SEM J2")
plt.grid(alpha=0.3); plt.legend(); plt.show()

# --------- i(ν): usa plot segmentado para evitar quebra no wrap 360->0
fig2, ax2 = plt.subplots()
_plot_i_vs_nu_segmentado(Nu_up_on,  I_up_on,  ax=ax2, color="#1f77b4", lw=1.2, label="VH Up  (J2 ON)")
_plot_i_vs_nu_segmentado(Nu_up_off, I_up_off, ax=ax2, color="#1f77b4", lw=1.2, label="VH Up  (J2 OFF)", alpha=0.35)
_plot_i_vs_nu_segmentado(Nu_v_on,   I_v_on,   ax=ax2, color="#2ca02c", lw=1.2, label="V Only (J2 ON)")
_plot_i_vs_nu_segmentado(Nu_v_off,  I_v_off,  ax=ax2, color="#2ca02c", lw=1.2, label="V Only (J2 OFF)", alpha=0.35)
_plot_i_vs_nu_segmentado(Nu_dn_on,  I_dn_on,  ax=ax2, color="#d62728", lw=1.2, label="VH Down (J2 ON)")
_plot_i_vs_nu_segmentado(Nu_dn_off, I_dn_off, ax=ax2, color="#d62728", lw=1.2, label="VH Down (J2 OFF)", alpha=0.35)
ax2.set_title("Inclinação i(ν): COM (opaco) × SEM J2 (transparente)")
ax2.legend(markerscale=4, fontsize=9)
plt.show()

# --------- Diferenças (ON – OFF): ΔΩ e Δi ao longo do tempo
plt.figure()
plt.plot(t_up,   _unwrap_deg(Om_up_on)  - _unwrap_deg(Om_up_off),  '-', color="#1f77b4", lw=1.2, label="ΔΩ VH Up")
plt.plot(t_v,    _unwrap_deg(Om_v_on)   - _unwrap_deg(Om_v_off),   '-', color="#2ca02c", lw=1.2, label="ΔΩ V Only")
plt.plot(t_down, _unwrap_deg(Om_dn_on)  - _unwrap_deg(Om_dn_off),  '-', color="#d62728", lw=1.2, label="ΔΩ VH Down")
plt.xlabel("Tempo [s]"); plt.ylabel("ΔΩ [graus]")
plt.title("Diferença de RAAN (J2 ON – J2 OFF)")
plt.grid(alpha=0.3); plt.legend(); plt.show()

plt.figure()
plt.plot(t_up,   I_up_on - I_up_off,   '-', color="#1f77b4", lw=1.2, label="Δi VH Up")
plt.plot(t_v,    I_v_on  - I_v_off,    '-', color="#2ca02c", lw=1.2, label="Δi V Only")
plt.plot(t_down, I_dn_on - I_dn_off,   '-', color="#d62728", lw=1.2, label="Δi VH Down")
plt.xlabel("Tempo [s]"); plt.ylabel("Δi [graus]")
plt.title("Diferença de Inclinação (J2 ON – J2 OFF)")
plt.grid(alpha=0.3); plt.legend(); plt.show()
