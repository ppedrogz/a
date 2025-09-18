import matplotlib.pyplot as plt
import numpy as np

import simulations.sat_vh_up as sat_vh_up
import simulations.Apogeu.sat_v_only as sat_v_only
import simulations.Apogeu.sat_vh_down as sat_vh_down

# Configuração: "first" para primeira órbita, "last" para última órbita

mu = 3.986e5  # km^3/s^2
def select_orbit(t, X, nus, incs, elems, mode: str):
  
    nus = np.asarray(nus)
    incs = np.asarray(incs)

    # "all" ou pouco dado: retorna tudo
    if mode == "all" or nus.size < 10:
        return t, X, nus, incs, elems

    # ---------- (A) DETECÇÃO ROBUSTA DE WRAPS EM ν ----------
    crossings = []
    for i in range(1, nus.size):
        if (nus[i-1] > 340.0) and (nus[i] < 20.0):
            crossings.append(i)
    crossings = np.array(crossings, dtype=int)

    def slice_between(i0, i1):
        i0 = int(max(0, min(i0, nus.size-2)))
        i1 = int(max(i0+1, min(i1, nus.size-1)))
        mask = np.zeros_like(nus, dtype=bool)
        mask[i0:i1+1] = True
        t_sel   = t[mask]
        X_sel   = X[:, mask]
        nus_sel = nus[mask]
        inc_sel = incs[mask]
        elems_sel = [elems[i] for i in range(len(elems)) if mask[i]]
        return t_sel, X_sel, nus_sel, inc_sel, elems_sel

    if crossings.size >= 2:
        if mode == "first":
            return slice_between(crossings[0], crossings[1])
        else:  # "last"
            return slice_between(crossings[-2], crossings[-1])

    if crossings.size == 1:
        if mode == "first":
            return slice_between(0, crossings[0])
        else:
            return slice_between(crossings[0], nus.size-1)

    # ---------- (B) FALLBACK POR PERÍODO (sempre fecha) ----------
    # Estado inicial:
    r0 = X[0:3, 0];  v0 = X[3:6, 0]
    r0n = float(np.linalg.norm(r0))
    v0n = float(np.linalg.norm(v0))
    # vis-viva: 1/a = 2/r - v^2/mu
    a0 = 1.0 / (2.0/r0n - (v0n**2)/mu)
    T_orb = 2.0*np.pi*np.sqrt(abs(a0)**3/mu)  # s

    if mode == "first":
        mask = (t - t[0]) <= T_orb
    else:  # "last"
        mask = (t[-1] - t) <= T_orb

    # garante quantidade mínima de pontos
    if mask.sum() < 10:
        k = max(10, int(0.01 * t.size))
        mask = np.zeros_like(mask, dtype=bool)
        if mode == "first":
            mask[:k] = True
        else:
            mask[-k:] = True

    t_sel   = t[mask]
    X_sel   = X[:, mask]
    nus_sel = nus[mask]
    inc_sel = incs[mask]
    elems_sel = [elems[i] for i in range(len(elems)) if mask[i]]
    return t_sel, X_sel, nus_sel, inc_sel, elems_sel

# Executa simulações
PLOT_MODE = "last"   # ou "last" ou "all"

print("Simulando satélite VH UP...")
res_up = sat_vh_up.simulate()
print("Simulando satélite V ONLY...")
res_v = sat_v_only.simulate()
print("Simulando satélite VH DOWN...")
res_down = sat_vh_down.simulate()

t_up, X_up, nus_up, incs_up, elems_up         = select_orbit(*res_up,   PLOT_MODE)
t_v, X_v, nus_v, incs_v, elems_v              = select_orbit(*res_v,    PLOT_MODE)
t_down, X_down, nus_down, incs_down, elems_dn = select_orbit(*res_down, PLOT_MODE)
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
