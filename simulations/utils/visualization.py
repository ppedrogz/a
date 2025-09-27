import matplotlib.pyplot as plt
from utils.types import OrbitalElements
import numpy as np
from cycler import cycler


def plot_classic_orbital_elements(t: np.typing.NDArray, orbital_elementss: list[OrbitalElements]):
    """
    Plots the classic orbital elements over time.

    Parameters:
    t (np.ndarray): Time array.
    orbital_elements (list): List of orbital elements objects.
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs[0, 0].plot(t, [element.major_axis for element in orbital_elementss], label='Major Axis')
    axs[0, 0].set_title('Major Axis')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Major Axis (km)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 1].plot(t, [element.eccentricity for element in orbital_elementss], label='Eccentricity', color='orange')
    axs[0, 1].set_title('Eccentricity')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Eccentricity')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    axs[1, 0].plot(t, [element.inclination for element in orbital_elementss], label='Inclination', color='green')
    axs[1, 0].set_title('Inclination')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Inclination (degrees)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[1, 1].plot(t, [element.ascending_node for element in orbital_elementss], label='Ascending Node', color='red')
    axs[1, 1].set_title('Ascending Node')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Ascending Node (degrees)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    axs[2, 0].plot(t, [element.argument_of_perigee for element in orbital_elementss], label='Argument of Perigee', color='purple')
    axs[2, 0].set_title('Argument of Perigee')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylabel('Argument of Perigee (degrees)')
    axs[2, 0].grid(True)
    axs[2, 0].legend()
    axs[2, 1].plot(t, [element.true_anomaly for element in orbital_elementss], label='True Anomaly', color='brown')
    axs[2, 1].set_title('True Anomaly')
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].set_ylabel('True Anomaly (degrees)')
    axs[2, 1].grid(True)
    axs[2, 1].legend()
    plt.tight_layout()
    plt.show()


def plot_classic_orbital_elements_overlay(*orbital_elementss_lists: list[np.typing.NDArray, list[OrbitalElements]]):
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    for orbital_elementss_list in orbital_elementss_lists:
        t = orbital_elementss_list[0]
        orbital_elementss = orbital_elementss_list[1]
        """
        Plots the classic orbital elements over time.

        Parameters:
        t (np.ndarray): Time array.
        orbital_elements (list): List of orbital elements objects.
        """
        axs[0, 0].plot(t, [element.major_axis for element in orbital_elementss], label='Major Axis')
        axs[0, 0].set_title('Major Axis')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Major Axis (km)')
        axs[0, 0].grid(True)
        axs[0, 0].legend()
        axs[0, 1].plot(t, [element.eccentricity for element in orbital_elementss], label='Eccentricity')
        axs[0, 1].set_title('Eccentricity')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Eccentricity')
        axs[0, 1].grid(True)
        axs[0, 1].legend()
        axs[1, 0].plot(t, [element.inclination for element in orbital_elementss], label='Inclination')
        axs[1, 0].set_title('Inclination')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Inclination (degrees)')
        axs[1, 0].grid(True)
        axs[1, 0].legend()
        axs[1, 1].plot(t, [element.ascending_node for element in orbital_elementss], label='Ascending Node')
        axs[1, 1].set_title('Ascending Node')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Ascending Node (degrees)')
        axs[1, 1].grid(True)
        axs[1, 1].legend()
        axs[2, 0].plot(t, [element.argument_of_perigee for element in orbital_elementss], label='Argument of Perigee')
        axs[2, 0].set_title('Argument of Perigee')
        axs[2, 0].set_xlabel('Time (s)')
        axs[2, 0].set_ylabel('Argument of Perigee (degrees)')
        axs[2, 0].grid(True)
        axs[2, 0].legend()
        axs[2, 1].plot(t, [element.true_anomaly for element in orbital_elementss], label='True Anomaly')
        axs[2, 1].set_title('True Anomaly')
        axs[2, 1].set_xlabel('Time (s)')
        axs[2, 1].set_ylabel('True Anomaly (degrees)')
    axs[2, 1].grid(True)
    axs[2, 1].legend()
    plt.tight_layout()
    plt.show()


def plot_3D_view(
        X,
        plot_earth: bool = True,
        earth_radius: float = 6378.0,
        earth_color: str = 'blue',
        earth_alpha: float = 0.3
        ):
    plt.figure()
    if plot_earth:
        ax = plt.axes(projection='3d')
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = earth_radius * np.cos(u)*np.sin(v)
        y = earth_radius * np.sin(u)*np.sin(v)
        z = earth_radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color=earth_color, alpha=earth_alpha)

    ax.plot3D(X[0, :], X[1, :], X[2, :], 'b-')
    ax.set_title('Orbit Propagation')
    ax.axis('equal')
    plt.show()

def plot_3D_overlay(
        *Xs,
        plot_earth: bool = True,
        earth_radius: float = 6378.0,
        earth_color: str = 'blue',
        earth_alpha: float = 0.3,
        orbit_marker: str = '-'
        ):
    plt.figure()
    plt.style.use('bmh')
    if plot_earth:
        ax = plt.axes(projection='3d')
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = earth_radius * np.cos(u)*np.sin(v)
        y = earth_radius * np.sin(u)*np.sin(v)
        z = earth_radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color=earth_color, alpha=earth_alpha)

    markers = ['-', '--', ':', '-.']
    for i in range(len(Xs)):
        X = Xs[i]
        ax.plot3D(X[0, :], X[1, :], X[2, :], markers[i], linewidth=3)
    ax.set_title('Orbit Propagation')
    ax.axis('equal')
    plt.show()

