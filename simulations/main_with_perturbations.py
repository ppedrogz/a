# main_constelacao.py  (exemplo robusto: chama perturbações só pelo main)
import numpy as np
import matplotlib.pyplot as plt

# seus módulos de simulação
import sat_vh_up as sat_vh_up
import sat_v_only as sat_v_only
import sat_vh_down as sat_vh_down


from perturbations import EarthParams, PerturbationFlags, accel_perturbations

# ========================= callback de perturbações =========================
# Unidades: km, s, km^3/s^2 -> retorno em km/s^2
P = EarthParams()                  # mu, Re, J2 padrão
FLAGS = PerturbationFlags(j2=True) 

def external_accel(r_vec: np.ndarray, v_vec: np.ndarray, t: float) -> np.ndarray:
    """
    Perturbações totais adicionadas pelo MAIN.
    Retorna (3,) km/s^2. Hoje: apenas J2.
    """
    return accel_perturbations(r_vec, v_vec, t, params=P, flags=FLAGS)

# ========================= helpers para invocação =========================
def _try_call_simulate(mod):
    """
    Tenta (1) passar o callback via kwargs;
          (2) injetar atributo 'external_accel' no módulo;
          (3) chamar puro e avisar se não deu para injetar.
    """
    # (1) tenta via kwargs (se simulate aceitar **kwargs)
    try:
        return mod.simulate(external_accel=external_accel)
    except TypeError:
        # (2) injeta atributo no módulo (se o código procurar por isso)
        try:
            setattr(mod, "external_accel", external_accel)
            return mod.simulate()
        except Exception as e:
            print(f"[WARN] {mod.__name__}: não foi possível injetar callback ({e}). Rodando sem perturbações.")
            # (3) roda sem perturbações
            return mod.simulate()

# ========================= roda as três simulações =========================
print("Simulando satélite VH UP...")
t_up, X_up, nus_up, incs_up, elems_up = _try_call_simulate(sat_vh_up)

print("Simulando satélite V ONLY...")
t_v, X_v, nus_v, incs_v, elems_v = _try_call_simulate(sat_v_only)

print("Simulando satélite VH DOWN...")
t_down, X_down, nus_down, incs_down, elems_down = _try_call_simulate(sat_vh_down)


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
