# arquivo: animacao_constelacao.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['animation.ffmpeg_path'] = r"C:\Users\ppggo\Downloads\ffmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"


from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necessário para 3D)

# === Importa os três modelos de satélites ===
import sat_v_only   as satV
import sat_vh_up    as satVHUp
import sat_vh_down  as satVHDown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# PARÂMETROS GERAIS (SÓ AJUSTAR AQUI)
# ============================================================

# Tempo total de integração [s]
DIAS_INTEGRACAO = 5        # número de dias que você quer integrar
TEMPO_FINAL_S   = DIAS_INTEGRACAO * 24 * 3600.0
   # ex.: 12h; teste rápido: 5400.0 (~1,5h)

# Número de pontos de tempo
# Número de pontos de tempo
N_PONTOS = 8000          # em vez de 2000

# Velocidade da animação
PASSO_FRAMES = 10        # em vez de 20 (menos pulo entre frames)
INTERVAL_MS  = 20        # pode manter assim, ou diminuir pra 10 se quiser mais rápido
               # mais pontos = órbita mais suave

# Perturbações globais
USAR_J2   = True
USAR_J22  = True
USAR_DRAG = True

# Velocidade da animação
         # intervalo entre frames (ms); menor = mais rápido


# ============================================================
# 1. Propagação usando os códigos existentes
# ============================================================

def propagar_satelites(t_final_s=TEMPO_FINAL_S, n_pontos=N_PONTOS,
                       usar_j2=USAR_J2, usar_j22=USAR_J22, usar_drag=USAR_DRAG):
    """
    Integra a dinâmica dos três satélites usando os módulos já existentes.

    Parâmetros
    ----------
    t_final_s : float
        Tempo total de integração (segundos).
    n_pontos : int
        Número de pontos de amostragem no tempo.
    usar_j2, usar_j22, usar_drag : bool
        Liga/desliga perturbações de achatamento (J2, J22) e arrasto.

    Retorna
    -------
    t : ndarray, shape (N,)
        Vetor de tempo comum às três simulações.
    rV, rVHup, rVHdown : ndarray, shape (N, 3)
        Vetores de posição (ECI, km) de cada satélite ao longo do tempo.
    """

    # Vetor de tempo comum (substitui o 't' interno de cada módulo)
    t = np.linspace(0.0, float(t_final_s), int(n_pontos))

    # --------------------------------------------------------
    # Satélite V-only
    # --------------------------------------------------------
    satV.t = t               # sobrescreve o vetor de integração interno
    satV._USE_J2   = usar_j2
    satV._USE_J22  = usar_j22
    satV._DRAG_ON  = usar_drag

    tV, XV, _, _, _ = satV.simulate()      # XV: [x,y,z,vx,vy,vz,m]

    # --------------------------------------------------------
    # Satélite V_H UP
    # --------------------------------------------------------
    satVHUp.t = t
    tUp, XUp, _, _, _ = satVHUp.simulate(j2=usar_j2,
                                         j22=usar_j22,
                                         drag=usar_drag)

    # --------------------------------------------------------
    # Satélite V_H DOWN
    # --------------------------------------------------------
    satVHDown.t = t
    tDown, XDown, _, _, _ = satVHDown.simulate(j2=usar_j2,
                                               j22=usar_j22,
                                               drag=usar_drag)

    # Garante que todos têm o mesmo tamanho de tempo
    assert len(tV) == len(tUp) == len(tDown) == len(t), \
        "As integrações retornaram vetores de tempo com tamanhos diferentes."

    # Posições (N,3)
    rV      = XV[0:3, :].T
    rVHup   = XUp[0:3, :].T
    rVHdown = XDown[0:3, :].T

    return t, rV, rVHup, rVHdown


# ============================================================
# 2. Desenho da Terra (com textura se 'earth.jpg' existir)
# ============================================================

def desenhar_terra(ax, raio_terra_km=6378.0):
    """
    Desenha a Terra como uma esfera no gráfico 3D.
    Tenta usar 'earth.jpg' ou 'earth.png' na MESMA pasta do arquivo .py.
    """

    # tenta jpg e png na pasta do script
    img = None
    for nome in ["earth.jpg", "earth.png"]:
        caminho = os.path.join(BASE_DIR, nome)
        if os.path.exists(caminho):
            img = plt.imread(caminho)
            break

    usar_textura = img is not None

    if usar_textura:
        ny, nx, _ = img.shape

        # longitude [-pi, pi], latitude [pi/2, -pi/2]
        lon = np.linspace(-np.pi, np.pi, nx)
        lat = np.linspace(np.pi/2, -np.pi/2, ny)
        lon, lat = np.meshgrid(lon, lat)

        x = raio_terra_km * np.cos(lat) * np.cos(lon)
        y = raio_terra_km * np.cos(lat) * np.sin(lon)
        z = raio_terra_km * np.sin(lat)

        ax.plot_surface(
            x, y, z,
            rstride=max(1, ny // 300),
            cstride=max(1, nx // 300),
            facecolors=img / 255.0,
            linewidth=0,
            antialiased=False
        )
    else:
        # Esfera simples caso não tenha textura
        u = np.linspace(0, 2*np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        x = raio_terra_km * np.outer(np.cos(u), np.sin(v))
        y = raio_terra_km * np.outer(np.sin(u), np.sin(v))
        z = raio_terra_km * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color="cornflowerblue", alpha=0.8)


def ajustar_aspecto_igual(ax, max_range):
    """
    Deixa o cubo de visualização com a mesma escala nos 3 eixos.
    """
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_box_aspect([1, 1, 1])


# ============================================================
# 3. Animação das órbitas
# ============================================================

def animar_orbitas(t, rV, rVHup, rVHdown,
                   salvar=False, nome_arquivo="orbitas.mp4",
                   passo_frames=PASSO_FRAMES, interval_ms=INTERVAL_MS):
    """
    Cria animação 3D das órbitas dos três satélites em torno da Terra.
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Terra
    desenhar_terra(ax, raio_terra_km=6378.0)

    # Limites
    r_norms = np.concatenate([
        np.linalg.norm(rV, axis=1),
        np.linalg.norm(rVHup, axis=1),
        np.linalg.norm(rVHdown, axis=1),
    ])
    max_r = 1.1 * np.max(r_norms)
    ajustar_aspecto_igual(ax, max_r)

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.set_title("Órbitas dos satélites em torno da Terra")

    # ============================================
    #   ✔️ CORES E ESTILOS ATUALIZADOS
    # ============================================

    # V-only → VERDE
    linha_V     = ax.plot([], [], [], lw=2, linestyle='-',
                          color="green", label="V-only")[0]
    ponto_V     = ax.plot([], [], [], "o", color="green", markersize=7)[0]

    # V_H UP → AZUL
    linha_VH_up = ax.plot([], [], [], lw=2, linestyle='-',
                          color="blue", label="V_H UP")[0]
    ponto_VH_up = ax.plot([], [], [], "s", color="blue", markersize=7)[0]

    # V_H DOWN → VERMELHO
    linha_VH_dn = ax.plot([], [], [], lw=2, linestyle='-',
                          color="red", label="V_H DOWN")[0]
    ponto_VH_dn = ax.plot([], [], [], "D", color="red", markersize=7)[0]

    ax.legend()

    # Frames acelerados
    N = len(t)
    indices_frames = list(range(1, N, passo_frames))

    def init():
        for ln in (linha_V, linha_VH_up, linha_VH_dn):
            ln.set_data([], [])
            ln.set_3d_properties([])
        for pt in (ponto_V, ponto_VH_up, ponto_VH_dn):
            pt.set_data([], [])
            pt.set_3d_properties([])
        return (linha_V, linha_VH_up, linha_VH_dn,
                ponto_V, ponto_VH_up, ponto_VH_dn)

    def update(frame_idx):
        k = indices_frames[frame_idx]

        # V-only
        linha_V.set_data(rV[:k, 0], rV[:k, 1])
        linha_V.set_3d_properties(rV[:k, 2])
        ponto_V.set_data([rV[k, 0]], [rV[k, 1]])
        ponto_V.set_3d_properties([rV[k, 2]])

        # V_H UP
        linha_VH_up.set_data(rVHup[:k, 0], rVHup[:k, 1])
        linha_VH_up.set_3d_properties(rVHup[:k, 2])
        ponto_VH_up.set_data([rVHup[k, 0]], [rVHup[k, 1]])
        ponto_VH_up.set_3d_properties([rVHup[k, 2]])

        # V_H DOWN
        linha_VH_dn.set_data(rVHdown[:k, 0], rVHdown[:k, 1])
        linha_VH_dn.set_3d_properties(rVHdown[:k, 2])
        ponto_VH_dn.set_data([rVHdown[k, 0]], [rVHdown[k, 1]])
        ponto_VH_dn.set_3d_properties([rVHdown[k, 2]])

        return (linha_V, linha_VH_up, linha_VH_dn,
                ponto_V, ponto_VH_up, ponto_VH_dn)

    ani = FuncAnimation(
        fig, update,
        frames=len(indices_frames),
        init_func=init,
        blit=False,
        interval=interval_ms
    )

    if salvar:
        ani.save(nome_arquivo, fps=30, dpi=150)
    else:
        plt.show()


# ============================================================
# 4. Bloco principal (para rodar no VS Code)
# ============================================================

if __name__ == "__main__":
    print("-> Integrando órbitas...")
    t, rV, rVHup, rVHdown = propagar_satelites()

    print("-> Gerando animação...")
    animar_orbitas(
        t, rV, rVHup, rVHdown,
        salvar= True,           # mude para True se quiser salvar em MP4
        nome_arquivo="orbitas.mp4"
    )
