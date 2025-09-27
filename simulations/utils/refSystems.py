import numpy as np
from dataclasses import dataclass
from utils.orbitalElements import OrbitalElements


def get_base_versors(state_vector):
    r_hat = np.linalg.norm(state_vector[0:3])
    v_hat = np.linalg.norm(state_vector[3:6])
    h_hat = np.linalg.norm(np.cross(state_vector[0:3], state_vector[3:6]))

    base = np.array([r_hat, v_hat, h_hat])

    return base


# FUNDAMENTAL ROTATIONS
def rot_X(angle_deg: float, direction: str = "clockwise") -> np.typing.NDArray:
    angle_rad = np.deg2rad(angle_deg)
    Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])

    if direction == "clockwise":
        Rx = Rx.transpose()
    elif direction == "counter-clockwise":
        Rx = Rx
    else:
        raise Exception("Invalid direction option, options are: clockwise, counter-clockwise")

    return Rx


def rot_Y(angle_deg: float, direction: str = "clockwise") -> np.typing.NDArray:
    angle_rad = np.deg2rad(angle_deg)
    Ry = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])

    if direction == "clockwise":
        Ry = Ry.transpose()
    elif direction == "counter-clockwise":
        Ry = Ry
    else:
        raise Exception("Invalid direction option, options are: clockwise, counter-clockwise")

    return Ry


def rot_Z(angle_deg: float, direction: str = "clockwise") -> np.typing.NDArray:
    angle_rad = np.deg2rad(angle_deg)
    Rz = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

    if direction == "clockwise":
        Rz = Rz.transpose()
    elif direction == "counter-clockwise":
        Rz = Rz
    else:
        raise Exception("Invalid direction option, options are: clockwise, counter-clockwise")

    return Rz


# SYSTEM REFERENCE ROTATIONS
def perifocal_to_inertial(points_perifocal: np.typing.NDArray, orbital_elements: OrbitalElements):
    omega = orbital_elements.argument_of_perigee
    Omega = orbital_elements.ascending_node
    i = orbital_elements.inclination

    print(omega, Omega, i)

    Rz_omega = rot_Z(omega, direction="counter-clockwise")
    Rx_i = rot_X(i, direction="counter-clockwise")
    Rz_Omega = rot_Z(Omega, direction="counter-clockwise")

    rotation_matrix = Rz_Omega @ Rx_i @ Rz_omega

    points_inertial = rotation_matrix @ points_perifocal  # @ é equivalente ao produto matricial

    return points_inertial


def inertial_to_perifocal(points_inertial: np.typing.NDArray, orbital_elements: OrbitalElements):
    omega = orbital_elements.argument_of_perigee
    Omega = orbital_elements.ascending_node
    i = orbital_elements.inclination

    Rz_omega = rot_Z(omega, direction="clockwise")
    Rx_i = rot_X(i, direction="clockwise")
    Rz_Omega = rot_Z(Omega, direction="clockwise")

    rotation_matrix = Rz_omega @ Rx_i @ Rz_Omega

    points_perifocal = rotation_matrix @ points_inertial

    return points_perifocal
