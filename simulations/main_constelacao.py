import matplotlib.pyplot as plt
import numpy as np

# importa cada propagador (cada um deve ter uma função simulate() que retorna resultados)
import sat_vh_up as sat_vh_up
import sat_v_only as sat_v_only
import sat_vh_down as sat_vh_down

# ----------------- helper: plot i(nu) segmentado, cor fixa -----------------
def plot_i_vs_nu_segmentado(nu_deg: np.ndarray, inc_deg: np.ndarray, *,
                            ax=None, color="black", label=None, **plot_kw):
    """
    Plota i(nu) em segmentos separados nos pontos de wrap (360->0),
    mantendo uma única cor para todos os segmentos.
    """
    nu_deg = np.asarray(nu_deg, dtype=float)
    inc_deg = np.asarray(inc_deg, dtype=float)
    if ax is None:
        fig, ax = plt.subplots()

    # Detecta wraps quando a série volta de ~360 para ~0
    dn = np.diff(nu_deg)
    wraps = np.where(dn < -180.0)[0]  # índice do ponto ANTES do wrap

    start = 0
    first_line = True
    for w in wraps:
        ax.plot(nu_deg[start:w+1], inc_deg[start:w+1], color=color,
                label=(label if first_line else None), **plot_kw)
        first_line = False
        start = w + 1
    # Último segmento
    ax.plot(nu_deg[start:], inc_deg[start:], color=color,
            label=(label if first_line else None), **plot_kw)

    ax.set_xlim(0.0, 360.0)
    ax.set_xlabel("Anomalia Verdadeira ν [graus]")
    ax.set_ylabel("Inclinação i [graus]")
    ax.grid(alpha=0.3)
    return ax

# ----------------- Executa as simulações (uma vez cada) -----------------
print("Simulando satélite VH UP...")
t_up, X_up, nus_up, incs_up, elems_up = sat_vh_up.simulate()

print("Simulando satélite V ONLY...")
t_v, X_v, nus_v, incs_v, elems_v = sat_v_only.simulate()

print("Simulando satélite VH DOWN...")
t_down, X_down, nus_down, incs_down, elems_down = sat_vh_down.simulate()

# ----------------- Gráfico 3D das órbitas -----------------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
earth_radius = 6378.0

u, vgrid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = earth_radius * np.cos(u)*np.sin(vgrid)
y_e = earth_radius * np.sin(u)*np.sin(vgrid)
z_e = earth_radius * np.cos(vgrid)
ax.plot_wireframe(x_e, y_e, z_e, color="g", alpha=0.3)
ax.set_box_aspect([1, 1, 1])

ax.plot3D(X_up[0, :], X_up[1, :], X_up[2, :], '-', color="#1f77b4", label="VH Up")
ax.plot3D(X_v[0, :], X_v[1, :], X_v[2, :],  '-', color="#2ca02c", label="V Only")
ax.plot3D(X_down[0, :], X_down[1, :], X_down[2, :], '-', color="#d62728", label="VH Down")

ax.set_title("Constelação - 3 Satélites")
ax.legend()
ax.axis('equal')
plt.show()

# ----------------- Inclinação vs anomalia verdadeira (3 séries, cor fixa) -----------------
fig2, ax2 = plt.subplots()
plot_i_vs_nu_segmentado(nus_up,   incs_up,   ax=ax2, color="#1f77b4", lw=1.2, label="VH Up")
plot_i_vs_nu_segmentado(nus_v,    incs_v,    ax=ax2, color="#2ca02c", lw=1.2, label="V Only")
plot_i_vs_nu_segmentado(nus_down, incs_down, ax=ax2, color="#d62728", lw=1.2, label="VH Down")
ax2.set_title("Inclinação vs ν - Comparação dos Satélites")
ax2.legend()
plt.show()

# ----------------- Massa vs tempo -----------------
plt.figure()
plt.plot(t_up,   X_up[6, :],   '-', color="#1f77b4", label="VH Up")
plt.plot(t_v,    X_v[6, :],    '-', color="#2ca02c", label="V Only")
plt.plot(t_down, X_down[6, :], '-', color="#d62728", label="VH Down")
plt.xlabel("Tempo [s]")
plt.ylabel("Massa [kg]")
plt.title("Consumo de Propelente")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
