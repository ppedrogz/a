# ==============================================================
# main_constelacao_last_orbit.py
# Objetivo: executar as três simulações (VH UP, V ONLY, VH DOWN),
# recortar APENAS a ÚLTIMA ÓRBITA COMPLETA pela FASE (u) e comparar
# inclinação vs ângulo orbital (usamos u, robusto para e≈0, i≈0).
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt

# Importa os propagadores (cada um deve expor simulate())
import sat_vh_up as sat_vh_up
import sat_v_only as sat_v_only
import sat_vh_down as sat_vh_down

MU_EARTH = 3.986e5  # km^3/s^2

# ----------------- Ângulo u (argumento da latitude) -----------------
def _arg_of_latitude_series_deg(X: np.ndarray) -> np.ndarray:
    """
    Retorna u (argumento da latitude) em graus, no intervalo [0, 360),
    para cada amostra de X=[r;v;...].
    Robusto para órbitas quase-circulares e quase-equatoriais.
    """
    k_hat = np.array([0.0, 0.0, 1.0])
    rM = X[0:3, :].T  # (N,3)
    vM = X[3:6, :].T  # (N,3)
    u_list = []
    for r_vec, v_vec in zip(rM, vM):
        h = np.cross(r_vec, v_vec); h_n = np.linalg.norm(h) + 1e-32
        h_hat = h / h_n
        n = np.cross(k_hat, h); n_n = np.linalg.norm(n)
        if n_n > 1e-12:
            p_hat = n / n_n
        else:
            # Equatorial: projeta î=(1,0,0) no plano orbital
            i_hat = np.array([1.0, 0.0, 0.0])
            p_tmp = i_hat - np.dot(i_hat, h_hat) * h_hat
            p_hat = p_tmp / (np.linalg.norm(p_tmp) + 1e-32)
        q_hat = np.cross(h_hat, p_hat)
        x = np.dot(r_vec, p_hat)
        y = np.dot(r_vec, q_hat)
        u_deg = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
        u_list.append(u_deg)
    return np.array(u_list, dtype=float)

def _unwrap_deg(a_deg: np.ndarray) -> np.ndarray:
    """Desembrulha ângulo em graus (equivalente ao np.unwrap em rad)."""
    a_rad = np.deg2rad(a_deg)
    a_unw = np.unwrap(a_rad)
    return np.rad2deg(a_unw)

# ----------------- Seleção do período orbital pela FASE (sem cortes) -----------------
def select_orbit_last_full_by_u(t, X, incs, elems):
    """
    Seleciona a ÚLTIMA órbita COMPLETA usando o argumento da latitude u (desembrulhado).
    Retorna: (t_sel, X_sel, u_sel_0_360, incs_sel, elems_sel)
    """
    t = np.asarray(t, float)
    incs = np.asarray(incs, float)
    N = X.shape[1]
    if not (t.size == N == incs.size == len(elems)):
        raise ValueError("Séries desalinhadas (t, X, incs, elems).")

    # 1) u(t) → unwrap para obter fase cumulativa
    u_deg = _arg_of_latitude_series_deg(X)
    u_unw = _unwrap_deg(u_deg)  # cresce monotonicamente (com pequenas oscilações permitidas)

    # 2) Encontra os índices onde a contagem de voltas incrementa (cada +360°)
    turns = np.floor(u_unw / 360.0).astype(int)
    jump_idx = np.where(np.diff(turns) >= 1)[0]  # índice "antes" do salto de volta

    if jump_idx.size >= 1:
        # Pegar a última volta completa: entre os DOIS últimos limites
        # Precisamos de pelo menos 2 limites; se tiver só 1, tentamos usar a penúltima por aproximação.
        if jump_idx.size >= 2:
            i0 = jump_idx[-2] + 1
            i1 = jump_idx[-1] + 1  # incluir o ponto do novo ciclo
        else:
            # Apenas um limite detectado: tenta formar uma volta com ~360° anteriores
            target = u_unw[jump_idx[-1] + 1] - 360.0
            # índice mais próximo do target
            i0 = int(np.argmin(np.abs(u_unw - target)))
            i1 = jump_idx[-1] + 1
            if i0 >= i1:
                i0 = max(0, i1 - (i1 // 10 + 10))
    else:
        # Falhou a detecção robusta: cai no período aproximado (vis-viva) no final
        rN = X[0:3, -1]; vN = X[3:6, -1]
        rNn = float(np.linalg.norm(rN)); vN2 = float(np.dot(vN, vN))
        aN = 1.0 / (2.0 / rNn - vN2 / MU_EARTH)
        T = 2.0 * np.pi * np.sqrt(abs(aN) ** 3 / MU_EARTH)
        t1 = t[-1]; t0 = t1 - T
        mask = (t >= t0) & (t <= t1)
        idx = np.where(mask)[0]
        i0, i1 = idx[0], idx[-1]

    # --- Fallback: se corte gerar menos de 100 amostras, usa período estimado via vis-viva
    slic = slice(i0, i1 + 1)
    if (i1 - i0) < 100:
        print("[WARN] Corte curto detectado — aplicando fallback temporal (vis-viva).")
        rN = X[0:3, -1]; vN = X[3:6, -1]
        rNn = float(np.linalg.norm(rN)); vN2 = float(np.dot(vN, vN))
        aN = 1.0 / (2.0 / rNn - vN2 / MU_EARTH)
        T = 2.0 * np.pi * np.sqrt(abs(aN) ** 3 / MU_EARTH)
        t1 = t[-1]; t0 = t1 - T
        mask = (t >= t0) & (t <= t1)
        idx = np.where(mask)[0]
        slic = slice(idx[0], idx[-1] + 1)

    # Fatiamento consistente
    t_sel     = t[slic]
    X_sel     = X[:, slic]
    incs_sel  = incs[slic]
    elems_sel = elems[slic.start:slic.stop]


    # 3) Recalcula u na janela e re-referencia para [0, 360) iniciando em zero (fase alinhada)
    u_win_unw = _unwrap_deg(_arg_of_latitude_series_deg(X_sel))
    u0 = u_win_unw[0]
    u_rel = (u_win_unw - u0) % 360.0  # 0..360
    # Garante que o último ponto seja 360°-ε (nunca exatamente 0, para não parecer "corte")
    if u_rel[-1] < 359.0:
        pass  # OK
    else:
        # pequena proteção numérica
        u_rel[-1] = 359.999

    # 4) Diagnóstico de fechamento geométrico (deve ser ~0 km)
    closure = np.linalg.norm(X_sel[0:3, 0] - X_sel[0:3, -1])
    print(f"[select_orbit] last full by u: slice={i0}:{i1}  |Δr_end-start|={closure:.6f} km (ideal ~ 0)")

    return t_sel, X_sel, u_rel, incs_sel, elems_sel

# ----------------- Execução das simulações -----------------
PLOT_MODE = "last"  # mantido para consistência; agora a seleção é sempre por fase (u)

# >>> Se desejar pular as simulações e carregar arquivos .npz, adapte aqui:
RUN_SIMULATIONS = True

if RUN_SIMULATIONS:
    print("Simulando satélite VH UP...")
    res_up = sat_vh_up.simulate()
    print("Simulando satélite V ONLY...")
    res_v  = sat_v_only.simulate()
    print("Simulando satélite VH DOWN...")
    res_dn = sat_vh_down.simulate()
else:
    # Exemplo de carga (ajuste nomes se já tiver salvo):
    up = np.load("output_vh_up.npz", allow_pickle=True)
    v  = np.load("output_v_only.npz", allow_pickle=True)
    dn = np.load("output_vh_down.npz", allow_pickle=True)
    res_up = (up["t"], up["X"], up["nus"], up["incs"], up["elems"].tolist())
    res_v  = (v["t"],  v["X"],  v["nus"],  v["incs"],  v["elems"].tolist())
    res_dn = (dn["t"], dn["X"], dn["nus"], dn["incs"], dn["elems"].tolist())

# Desempacota
t_up,   X_up,   nus_up,   incs_up,   elems_up  = res_up
t_v,    X_v,    nus_v,    incs_v,    elems_v   = res_v
t_down, X_down, nus_down, incs_down, elems_dn  = res_dn

# --- Recorta a ÚLTIMA órbita COMPLETA pela fase u (sem cortes) ---
t_up,   X_up,   u_up,   incs_up,   elems_up  = select_orbit_last_full_by_u(t_up,   X_up,   incs_up,   elems_up)
t_v,    X_v,    u_v,    incs_v,    elems_v   = select_orbit_last_full_by_u(t_v,    X_v,    incs_v,    elems_v)
t_down, X_down, u_down, incs_down, elems_dn  = select_orbit_last_full_by_u(t_down, X_down, incs_down, elems_dn)

# ----------------- Plot 3D da órbita (volta completa, sem mordida) -----------------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
earth_radius = 6378.0

ugrid, vgrid = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_e = earth_radius * np.cos(ugrid) * np.sin(vgrid)
y_e = earth_radius * np.sin(ugrid) * np.sin(vgrid)
z_e = earth_radius * np.cos(vgrid)
ax.plot_wireframe(x_e, y_e, z_e, color="g", alpha=0.3)
ax.set_box_aspect([1, 1, 1])

ax.plot3D(X_up[0, :],   X_up[1, :],   X_up[2, :],   'b-', label="VH Up")
ax.plot3D(X_v[0, :],    X_v[1, :],    X_v[2, :],    'g-', label="V Only")
ax.plot3D(X_down[0, :], X_down[1, :], X_down[2, :], 'r-', label="VH Down")

ax.set_title(f"Constelação - 3 Satélites (última órbita completa por fase u)")
ax.legend()
ax.axis('equal')
plt.show()

# ----------------- Inclinação vs ângulo orbital (usamos u re-referenciado) -----------------
plt.figure()
plt.plot(u_up,   incs_up,   'b.', ms=0.9, label="VH Up")
plt.plot(u_v,    incs_v,    'g.', ms=0.9, label="V Only")
plt.plot(u_down, incs_down, 'r.', ms=0.9, label="VH Down")
plt.xlim(0, 360)
plt.xlabel("Ângulo orbital u (graus)")  # robusto p/ circulares
plt.ylabel("Inclinação i (graus)")
plt.title("Inclinação vs u — última órbita completa (sem cortes)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
