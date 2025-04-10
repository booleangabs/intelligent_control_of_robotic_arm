import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import pandas as pd
from tqdm import tqdm
import cv2

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

def pixel_to_screen(x_img, y_img, h, w):
    ox, oy = w // 2, h // 2
    x_ndc = x_img / w
    y_ndc = 1 - (y_img / h)
    x_screen = (2 * x_ndc - 1)
    y_screen = (2 * y_ndc - 1)
    return x_screen, y_screen


lengths = list(i / 100 for i in [10, 12.4, 6])
z = 10

theta_series = np.deg2rad(np.linspace(10, 170, 5) - 90)
theta_array = np.float32([
    [t1, t2, t3, t4, 0, 0]
    for t1 in np.deg2rad(np.linspace(15, 165, 8) - 90)
    for t2 in theta_series
    for t3 in theta_series
    for t4 in theta_series
])

reconstructed_points = np.float32(
    list(fkine(t, lengths, z / 100) for t in theta_array)
)

mask = reconstructed_points[..., 0] >= 0

theta_array = theta_array[mask]
reconstructed_points = reconstructed_points[mask]

radius_constraint1 = np.sqrt(reconstructed_points[..., 0]**2 + reconstructed_points[..., 1]**2) >= 0.125
radius_constraint2 = np.sqrt(reconstructed_points[..., 0]**2 + reconstructed_points[..., 1]**2) <= 0.2725
mask = np.bitwise_and(radius_constraint1, radius_constraint2)

theta_array = theta_array[mask]
reconstructed_points = reconstructed_points[mask]

client = RemoteAPIClient()
sim = client.require('sim')

sim.setStepping(True)

sim.startSimulation()
time.sleep(2)
try:
    webcam_handle = sim.getObject("./webcam")
    target = sim.getObject("./Target")
    motors = list([sim.getObject(f"./Motor{i}") for i in range(1, 4 + 1)])
    data = {
        "motor_0": [], 
        "motor_1": [],
        "motor_2": [], 
        "motor_3": [],
        "x_img": [], 
        "y_img": [], 
        "x_screen": [], 
        "y_screen": []
    }
    for k in tqdm(range(len(theta_array))):
        sim.setObjectPosition(target, reconstructed_points[k])
        ori = np.clip(np.arctan2(reconstructed_points[k][1], reconstructed_points[k][0] - 0.04), -np.pi / 2, np.pi / 2)
        sim.setObjectOrientation(target, [0, 0, ori])
        for _ in range(5):
            sim.step()
        time.sleep(1.5)
        img, res = sim.getVisionSensorImg(webcam_handle)
        img = np.reshape(sim.unpackUInt8Table(img), res[::-1] + [3])
        img = np.flipud(img)[..., ::-1].astype("uint8")

        # HSV magenta detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_magenta = np.array([140, 100, 100])
        upper_magenta = np.array([165, 255, 255])
        mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Only log centroid after the full delay has passed
        centroid = None
        contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroid = (cx, cy)
                    contour = cnt
                    break  # Only take first valid detection

        # Annotate & show image
        if centroid:
            cx, cy = centroid
            # Plane is the magenta marker in our simulation
            # I do know this is kind of cheating but we might not need to use this for anything
            # other than filtering points by "z"
            x_screen, y_screen = pixel_to_screen(cx, cy, h=img.shape[0], w=img.shape[1])
            data["x_img"].append(cx)
            data["y_img"].append(cy)
            data["x_screen"].append(x_screen)
            data["y_screen"].append(y_screen)
            for l, motor in enumerate(motors):
                data[f"motor_{l}"].append(sim.getJointPosition(motor))
        df = pd.DataFrame(data)
        df.to_csv(f"robot_arm_cam{img.shape[1]}x{img.shape[0]}.csv", index=False)
except Exception as e:
    sim.stopSimulation()
    raise e
finally:
    df = pd.DataFrame(data)
    df.to_csv(f"robot_arm_cam{img.shape[1]}x{img.shape[0]}.csv", index=False)
    sim.stopSimulation()


