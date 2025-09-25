import matplotlib.pyplot as plt
import numpy as np

import sat_vh_up as sat_vh_up;
import sat_v_only as sat_v_only;
import sat_vh_down as sat_vh_down;
# Configuração: "first" para primeira órbita, "last" para última órbita
mu = 3.986e5  # km^3/s^2
import numpy as np

def _arg_of_latitude_series(X: np.ndarray) -> np.ndarray:
    """u em [0,360): robusto mesmo para e≈0."""
    k_hat = np.array([0.0, 0.0, 1.0])
    rM = X[0:3, :].T
    vM = X[3:6, :].T
    u_list = []
    for r_vec, v_vec in zip(rM, vM):
        h = np.cross(r_vec, v_vec); h_n = np.linalg.norm(h) + 1e-32
        h_hat = h / h_n
        n = np.cross(k_hat, h); n_n = np.linalg.norm(n)
        if n_n > 1e-12:
            p_hat = n / n_n
        else:
            i_hat = np.array([1.0, 0.0, 0.0])
            p_tmp = i_hat - np.dot(i_hat, h_hat)*h_hat
            p_hat = p_tmp / (np.linalg.norm(p_tmp) + 1e-32)
        q_hat = np.cross(h_hat, p_hat)
        x = np.dot(r_vec, p_hat); y = np.dot(r_vec, q_hat)
        u_list.append((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)
    return np.array(u_list, float)

def select_orbit(t, X, nus, incs, elems, mode: str):
    """
    Retorna uma órbita COMPLETA:
      - first: entre os 2 primeiros wraps de u
      - last:  entre os 2 últimos wraps de u
      - all:   série inteira
    """
    t    = np.asarray(t, float)
    nus  = np.asarray(nus, float)
    incs = np.asarray(incs, float)
    N = X.shape[1]
    if not (t.size == N == nus.size == incs.size == len(elems)):
        raise ValueError("Séries desalinhadas (t, X, nus, incs, elems).")

    if mode == "all" or N < 10:
        return t, X, nus, incs, elems

    # ---- wraps por argumento de latitude u (nó ascendente) ----
    u = _arg_of_latitude_series(X)
    dn = np.diff(u)
    wraps = np.where(dn < -180.0)[0]          # índice ANTES do wrap 360->0

    # ---- escolher pares de wraps para órbita completa ----
    if wraps.size >= 2:
        if mode == "first":
            i0 = wraps[0] + 1   # começa imediatamente APÓS o 1º wrap
            i1 = wraps[1]       # e termina no 2º wrap
        elif mode == "last":
            i0 = wraps[-2] + 1  # começa após o penúltimo wrap
            i1 = wraps[-1]      # termina no último wrap
        else:
            raise ValueError("mode deve ser 'first' | 'last' | 'all'")
        slic = slice(i0, i1 + 1)
    else:
        # Fallback: 1 período aproximado (vis-viva) no início/fim
        r0 = X[0:3, 0]; v0 = X[3:6, 0]
        r0n = float(np.linalg.norm(r0)); v02 = float(np.dot(v0, v0))
        mu = 3.986e5  # km^3/s^2 (ou passe como arg se preferir)
        a0 = 1.0 / (2.0/r0n - v02/mu)
        T_orb = 2.0*np.pi*np.sqrt(abs(a0)**3/mu)
        if mode == "first":
            mask = (t - t[0]) <= T_orb
        else:
            mask = (t[-1] - t) <= T_orb
        idx = np.where(mask)[0]
        if idx.size < 10:  # garante pontos
            k = max(10, int(0.01 * t.size))
            idx = np.arange(0, k) if mode == "first" else np.arange(t.size-k, t.size)
        slic = slice(idx[0], idx[-1] + 1)

    # ---- fatiamento consistente ----
    t_sel     = t[slic]
    X_sel     = X[:, slic]
    nus_sel   = nus[slic]
    incs_sel  = incs[slic]
    elems_sel = elems[slic.start:slic.stop]

    # ---- checagem de fechamento (debug) ----
    r_start = X_sel[0:3, 0]; r_end = X_sel[0:3, -1]
    closure = np.linalg.norm(r_end - r_start)
    print(f"[select_orbit] wraps={wraps.size}, slice={slic.start}:{slic.stop-1}, "
          f"|r_end-r_start|={closure:.6f} km (ideal ~ 0)")
    return t_sel, X_sel, nus_sel, incs_sel, elems_sel

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
ax.axis('equal')
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
