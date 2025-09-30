# utils/angles.py
from __future__ import annotations
import numpy as np

DEG2RAD = np.pi/180.0
RAD2DEG = 180.0/np.pi

def wrap_to_2pi(theta_rad: np.ndarray | float) -> np.ndarray | float:
    return np.mod(theta_rad, 2.0*np.pi)

def wrap_to_pi(theta_rad: np.ndarray | float) -> np.ndarray | float:
    th = np.mod(theta_rad + np.pi, 2.0*np.pi)
    th = np.where(th < 0.0, th + 2.0*np.pi, th)
    return th - np.pi

def contiguify_from_prev(angles_rad: np.ndarray) -> np.ndarray:
    """Gera série contínua (sem saltos 2π) a partir de uma série em [0,2π)."""
    a = np.array(angles_rad, dtype=float).copy()
    for k in range(1, len(a)):
        da = a[k] - a[k-1]
        if da >  np.pi: a[k:] -= 2.0*np.pi
        if da < -np.pi: a[k:] += 2.0*np.pi
    return a

def rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: 
        return x
    win = int(win) | 1
    pad = win//2
    xp = np.pad(x, (pad, pad), mode='edge')
    c = np.convolve(xp, np.ones(win)/win, mode='valid')
    return c

def deg(xrad: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(xrad)*RAD2DEG

def rad(xdeg: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(xdeg)*DEG2RAD
