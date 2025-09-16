import matplotlib.pyplot as plt
import numpy as np

import sat_vh_up
import sat_v_only
import sat_vh_down

# Configuração: "first" para primeira órbita, "last" para última órbita
PLOT_MODE = "first"   # ou "last"

def select_orbit(t, X, nus, incs, elems, earth_radius=6378.0):
    """Extrai apenas a primeira ou última órbita do resultado."""
    # calcula período orbital médio inicial
    a0 = elems[0][0]  # semi-eixo maior inicial
    mu = 3.986e5
    T_orb = 2*np.pi*np.sqrt(a0**3/mu)

    if PLOT_MODE == "first":
        mask = t <= T_orb
    elif PLOT_MODE == "last":
        mask = t >= (t[-1] - T_orb)
    else:
        raise ValueError("PLOT_MODE deve ser 'first' ou 'last'")

    return t[mask], X[:, mask], np.array(nus)[mask], np.array(incs)[mask], [elems[k] for k in range(len(mask)) if mask[k]]

# Executa simulações
print("Simulando satélites...")
res_up   = sat_vh_up.simulate()
res_v    = sat_v_only.simulate()
res_down = sat_vh_down.simulate()

# Seleciona apenas a órbita desejada
t_up, X_up, nus_up, incs_up, elems_up     = select_orbit(*res_up)
t_v, X_v, nus_v, incs_v, elems_v          = select_orbit(*res_v)
t_down, X_down, nus_down, incs_down, elems_down = select_orbit(*res_down)

# ---------- Plot 3D ----------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
earth_radius = 6378.0

u, vgrid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = earth_radius * np.cos(u)*np.sin(vgrid)
y_e = earth_radius * np.sin(u)*np.sin(vgrid)
z_e = earth_radius * np.cos(vgrid)
ax.plot_wireframe(x_e, y_e, z_e, color="g", alpha=0.3)
ax.set_box_aspect([1,1,1])

ax.plot3D(X_up[0, :], X_up[1, :], X_up[2, :], 'b-', label="VH Up")
ax.plot3D(X_v[0, :], X_v[1, :], X_v[2, :], 'g-', label="V Only")
ax.plot3D(X_down[0, :], X_down[1, :], X_down[2, :], 'r-', label="VH Down")

ax.set_title(f"Constelação - 3 Satélites ({PLOT_MODE} orbit)")
ax.legend()
plt.show()

# Inclinação vs anomalia verdadeira
plt.figure()
plt.plot(nus_up, incs_up, 'b.', ms=0.7, label="VH Up")
plt.plot(nus_v, incs_v, 'g.', ms=0.7, label="V Only")
plt.plot(nus_down, incs_down, 'r.', ms=0.7, label="VH Down")
plt.xlim(0, 360)
plt.xlabel("Anomalia Verdadeira ν [graus]")
plt.ylabel("Inclinação i [graus]")
plt.title(f"Inclinação vs ν ({PLOT_MODE} orbit)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
