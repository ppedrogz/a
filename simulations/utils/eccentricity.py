# utils/ecc_plot.py
import numpy as np
import matplotlib.pyplot as plt

def eccentricity_vector_from_rv(r: np.ndarray, v: np.ndarray, mu: float) -> np.ndarray:
    """
    Vetor excentricidade: e = (v x h)/mu - r/|r|
    r, v em km e km/s; mu em km^3/s^2. Retorna (3,)
    """
    r = np.asarray(r, float).reshape(3)
    v = np.asarray(v, float).reshape(3)
    h = np.cross(r, v)
    e_vec = (np.cross(v, h) / mu) - (r / (np.linalg.norm(r) + 1e-32))
    return e_vec

def eccentricity_series_from_X(X: np.ndarray, mu: float) -> np.ndarray:
    """
    X: matriz de estados 6xN ou 7xN (linhas [r; v; (m)]). Retorna matriz Nx3 com e_vec[k].
    """
    rM = X[0:3, :].T
    vM = X[3:6, :].T
    e_list = [eccentricity_vector_from_rv(r, v, mu) for r, v in zip(rM, vM)]
    return np.vstack(e_list)  # shape (N,3)

def plot_eccentricity_time(t: np.ndarray, X: np.ndarray, mu: float):
    """
    Plota ex, ey, ez e |e| ao longo do tempo.
    """
    eM = eccentricity_series_from_X(X, mu)      # (N,3)
    e_norm = np.linalg.norm(eM, axis=1)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(t, eM[:,0], label='e_x')
    axs[0].plot(t, eM[:,1], label='e_y')
    axs[0].plot(t, eM[:,2], label='e_z')
    axs[0].set_ylabel('Componentes de e [-]')
    axs[0].grid(True); axs[0].legend()

    axs[1].plot(t, e_norm, color='k', lw=1.75, label='|e|')
    axs[1].set_xlabel('Tempo [s]')
    axs[1].set_ylabel('|e| [-]')
    axs[1].grid(True); axs[1].legend()

    fig.suptitle('Vetor excentricidade — componentes e módulo')
    fig.tight_layout()
    return fig, axs

def plot_eccentricity_in_orbital_plane(X: np.ndarray, mu: float, title='Trajetória de e no plano orbital'):
    """
    Projeta e(t) no plano orbital inicial (perifocal) e plota a trajetória.
    Útil para ver rotação de ω: para e>0, o vetor aponta para o perigeu.
    """
    r0, v0 = X[0:3, 0], X[3:6, 0]
    h0 = np.cross(r0, v0); h0n = np.linalg.norm(h0) + 1e-32
    h_hat = h0 / h0n

    # eixo p̂ inicial: se e≈0, use direção do nodo ascendente
    e0 = eccentricity_vector_from_rv(r0, v0, mu)
    e0n = np.linalg.norm(e0)
    if e0n > 1e-10:
        p_hat = e0 / e0n
    else:
        k_hat = np.array([0.0, 0.0, 1.0])
        n_vec = np.cross(k_hat, h_hat)
        if np.linalg.norm(n_vec) < 1e-10:     # equatorial pura → escolha eixo arbitrário no plano
            i_hat = np.array([1.0, 0.0, 0.0])
            n_vec = i_hat - np.dot(i_hat, h_hat)*h_hat
        p_hat = n_vec / (np.linalg.norm(n_vec) + 1e-32)

    q_hat = np.cross(h_hat, p_hat)

    # série e(t) e projeções no plano perifocal
    eM = eccentricity_series_from_X(X, mu)   # (N,3)
    ep = eM @ p_hat    # componente ao longo de p̂
    eq = eM @ q_hat    # componente ao longo de q̂

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ep, eq, '-', color='tab:red', lw=1.5)
    ax.scatter(ep[0],  eq[0],  c='g', s=40, label='início')
    ax.scatter(ep[-1], eq[-1], c='k', s=40, label='fim')
    ax.axhline(0, color='0.7', lw=0.8); ax.axvline(0, color='0.7', lw=0.8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$e_p$ (ao longo de $\hat p$)'); ax.set_ylabel(r'$e_q$ (ao longo de $\hat q$)')
    ax.set_title(title); ax.grid(True); ax.legend()
    return fig, ax
