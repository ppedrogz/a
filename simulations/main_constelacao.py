import matplotlib.pyplot as plt
import numpy as np

# importa cada propagador (cada um deve ter uma função simulate() que retorna resultados)
import sat_vh_up as sat_vh_up
import sat_v_only as sat_v_only
import sat_vh_down as sat_vh_down

# Executa as simulações
print("Simulando satélite VH UP...")
res_up = sat_vh_up.simulate()

print("Simulando satélite V ONLY...")
res_v = sat_v_only.simulate()

print("Simulando satélite VH DOWN...")
res_down = sat_vh_down.simulate()

# unpack dos resultados
t_up, X_up, nus_up, incs_up, elems_up = res_up
t_v, X_v, nus_v, incs_v, elems_v = res_v
t_down, X_down, nus_down, incs_down, elems_down = res_down
# ----------------- Gráficos comparativos -----------------
t_up, X_up, nus_up, incs_up, elems_up = sat_vh_up.simulate()
t_v, X_v, nus_v, incs_v, elems_v = sat_v_only.simulate()
t_down, X_down, nus_down, incs_down, elems_down = sat_vh_down.simulate()


# Plot 3D das órbitas
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
earth_radius = 6378.0

u, vgrid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = earth_radius * np.cos(u)*np.sin(vgrid)
y_e = earth_radius * np.sin(u)*np.sin(vgrid)
z_e = earth_radius * np.cos(vgrid)
ax.plot_wireframe(x_e, y_e, z_e, color="g")
ax.set_box_aspect([1, 1, 1])  

ax.plot3D(X_up[0, :], X_up[1, :], X_up[2, :], 'b-', label="VH Up")
ax.plot3D(X_v[0, :], X_v[1, :], X_v[2, :], 'g-', label="V Only")
ax.plot3D(X_down[0, :], X_down[1, :], X_down[2, :], 'r-', label="VH Down")

ax.set_title("Constelação - 3 Satélites")
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
plt.title("Inclinação vs ν - Comparação dos Satélites")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Massa vs tempo
plt.figure()
plt.plot(t_up, X_up[6, :], 'b-', label="VH Up")
plt.plot(t_v, X_v[6, :], 'g-', label="V Only")
plt.plot(t_down, X_down[6, :], 'r-', label="VH Down")
plt.xlabel("Tempo [s]")
plt.ylabel("Massa [kg]")
plt.title("Consumo de Propelente")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
