import numpy as np


def map_theta(target_theta, current_theta, directions=np.float32([1, 1, -1, 1, 1, 1])):
    dtheta = directions * (target_theta - current_theta)
    return np.rad2deg(dtheta) + np.float32([80, 80, 50, 50, 0, 0])