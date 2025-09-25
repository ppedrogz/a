import matplotlib.pyplot as plt
import numpy as np

#import satvup as sat_vh_up
#import satv as sat_v_only
#import satdown as sat_vh_down
import sat_vh_up as sat_vh_up
import sat_v_only as sat_v_only
import sat_vh_down as sat_vh_down
# Configuração: "first" para primeira órbita, "last" para última órbita

mu = 3.986e5  # km^3/s^2
import numpy as np

def select_orbit(t, X, nus, incs, elems, mode: str):
    """
    Seleciona uma órbita completa com base nos 'wraps' da anomalia verdadeira ν.
    mode: "first" | "last" | "all"
    Retorna tupla (t_sel, X_sel, nus_sel, incs_sel, elems_sel)
    """
    # --- garantias de alinhamento ---
    t   = np.asarray(t,   dtype=float)
    nus = np.asarray(nus, dtype=float)
    incs = np.asarray(incs, dtype=float)

    N = X.shape[1]
    if not (t.size == N == nus.size == incs.size == len(elems)):
        raise ValueError(f"Tamanhos incompatíveis: len(t)={t.size}, N={N}, "
                         f"len(nus)={nus.size}, len(incs)={incs.size}, len(elems)={len(elems)}")

    # "all": retorna tudo diretamente
    if mode == "all" or N == 0:
        return t, X, nus, incs, elems

    # --- detecção de wraps (pontos ANTES do salto 360→0) ---
    # dn < -180 detecta quedas grandes (ex.: 359 -> 1)
    dn = np.diff(nus)
    wraps = np.where(dn < -180.0)[0]  # índice do ponto ANTES do wrap

    # --- construir segmentos [i0, i1] inteiros de órbitas completas ---
    if wraps.size == 0:
        # Sem wrap detectado: considere toda a série como uma "órbita"
        segments = [(0, N - 1)]
    else:
        starts = np.r_[0, wraps + 1]   # começo após cada wrap
        ends   = np.r_[wraps, N - 1]   # termina no ponto do wrap e no fim
        segments = list(zip(starts, ends))

        # Filtrar segmentos degenerados (i0 <= i1)
        segments = [(i0, i1) for (i0, i1) in segments if (0 <= i0 < N and 0 <= i1 < N and i0 <= i1)]
        if not segments:
            segments = [(0, N - 1)]

    # --- escolher segmento conforme o mode ---
    if mode == "first":
        i0, i1 = segments[0]
    elif mode == "last":
        i0, i1 = segments[-1]
    else:
        raise ValueError("mode deve ser 'first', 'last' ou 'all'")

    # --- fatia por índices inteiros (evita mask de tamanho errado) ---
    slic = slice(i0, i1 + 1)
    t_sel    = t[slic]
    X_sel    = X[:, slic]
    nus_sel  = nus[slic]
    incs_sel = incs[slic]
    elems_sel = elems[i0:i1+1]

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
