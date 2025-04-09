import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def fkine(theta: np.ndarray, lenghts: list, z: float) -> np.ndarray:
    """Forward kinematics for robotic arm

    Args:
        theta (np.ndarray): Motor angles
        lenghts (list): First three stages lengths
        z (float): Fixed height of actuator

    Returns:
        np.ndarray: XYZ position w.r.t. to base motor
    """
    t1, t2, t3, t4, *_ = theta
    l1, l2, l3 = lenghts

    d = (l1 * np.cos(t2) \
            + l2 * np.cos(t2 + t3)) \
            + l3 * np.cos(t2 + t3 + t4)

    x = np.cos(t1) * d
    y = np.sin(t1) * d
    position = np.float32([x, y, z])
    return position

def ikine(position: np.ndarray, lenghts: list) -> np.ndarray:
    """Inverse kinematics for robotic arm

    Args:
        position (np.ndarray): XYZ position w.r.t. to base motor
        lenghts (list): First three stages lengths

    Returns:
        np.ndarray: Motor angles
    """
    x, y, z = position
    l1, l2, l3 = lenghts

    # phi = np.atan2(z, abs(y))  # Actual calculation
    phi = np.deg2rad(10)

    cphi, sphi = np.cos(phi), np.sin(phi)

    t1 = np.arctan2(y, x)

    new_x = np.sqrt(x**2 + y**2)
    x2 = new_x - l3 * abs(cphi)
    z2 = z - l3 * sphi

    c3 = (x2**2 + z2**2 - l1**2 - l2**2) / (2 * l1 * l2)

    t3 = -np.arccos(np.clip(c3, -1, 1))
    s3 = np.sin(t3)

    k1 = l1 + l2 * c3
    k2 = l2 * s3
    k3 = x2**2 + z2**2
    c2 = (k1 * x2 + k2 * z2) / k3
    s2 = (k1 * z2 - k2 * x2) / k3

    t2 = np.arctan2(s2, c2)
    t4 = phi - (t2 + t3)

    theta = np.float32([t1, t2, t3, t4, 0, 0])
    return theta

lengths = list(i / 100 for i in [10, 12.4, 6])
z = 10

x_series = np.linspace(-0.3, 0.3, 35)
y_series = np.linspace(0, 0.3, 35)
coords = np.float32([
    [x, y, z / 100] for x in x_series for y in y_series
])

theta_array = np.float32(
    list(ikine(c, lenghts=lengths) for c in coords)
)

reconstructed_points = np.float32(
    list(fkine(t, lengths, z / 100) for t in theta_array)
)

valid_mask = np.sum(np.abs(reconstructed_points - coords) < 1e-4, axis=1) == 3
radius_constraint = np.sqrt(coords[..., 0]**2 + coords[..., 1]**2) >= 0.125
mask = np.bitwise_and(valid_mask, radius_constraint)

print(f"N. valid points: {mask.sum()}")

plt.scatter(coords[..., 0], coords[..., 1], c=mask, label="Invalid positions", s=50, marker="*", cmap="viridis")
plt.legend()
plt.gca().set_aspect("equal")
plt.show()