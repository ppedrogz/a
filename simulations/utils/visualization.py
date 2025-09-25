import numpy as np
import matplotlib.pyplot as plt

def plot_classic_orbital_elements(t, orbital_elementss):
    """
    Plots the classic orbital elements over time.

    Parameters:
    t (np.ndarray): Time array.
    orbital_elementss (list): List of orbital elements objects.
    """

    # helper local: desfaz saltos de 360° para série contínua
    def _unwrap_deg(a_deg):
        a_deg = np.asarray(a_deg, dtype=float)
        return np.degrees(np.unwrap(np.radians(a_deg)))

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # a) semi-eixo maior
    axs[0, 0].plot(t, [el.major_axis for el in orbital_elementss], label='Major Axis')
    axs[0, 0].set_title('Major Axis')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Major Axis (km)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # b) excentricidade
    axs[0, 1].plot(t, [el.eccentricity for el in orbital_elementss], label='Eccentricity', color='orange')
    axs[0, 1].set_title('Eccentricity')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Eccentricity')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # c) inclinação
    axs[1, 0].plot(t, [el.inclination for el in orbital_elementss], label='Inclination', color='green')
    axs[1, 0].set_title('Inclination')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Inclination (degrees)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # d) RAAN (Ω) — unwrapped para evitar “paredões” 360→0
    Omega_series = [el.ascending_node for el in orbital_elementss]
    axs[1, 1].plot(t, _unwrap_deg(Omega_series), label='Ascending Node (unwrap)', color='red')
    axs[1, 1].set_title('Ascending Node')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Ascending Node (degrees)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # e) argumento do perigeu
    axs[2, 0].plot(t, [el.argument_of_perigee for el in orbital_elementss],
                   label='Argument of Perigee', color='purple')
    axs[2, 0].set_title('Argument of Perigee')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylabel('Argument of Perigee (degrees)')
    axs[2, 0].grid(True)
    axs[2, 0].legend()

    # f) anomalia verdadeira
    axs[2, 1].plot(t, [el.true_anomaly for el in orbital_elementss], label='True Anomaly', color='brown')
    axs[2, 1].set_title('True Anomaly')
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].set_ylabel('True Anomaly (degrees)')
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()
