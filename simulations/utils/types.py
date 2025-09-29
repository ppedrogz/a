from dataclasses import dataclass


@dataclass
class OrbitalElements:
    major_axis: float
    eccentricity: float
    inclination: float
    ascending_node: float   # Right ascension of the ascending node
    argument_of_perigee: float
    true_anomaly: float
    argument_of_latitude: float
