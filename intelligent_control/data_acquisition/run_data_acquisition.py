import matplotlib.pyplot as plt
import pandas as pd
import serial
import time
import numpy as np
import cv2
import cv2.aruco as aruco

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

def map_theta(target_theta, current_theta, directions=np.float32([1, 1, -1, 1, 1, 1])):
    dtheta = directions * (target_theta - current_theta)
    return np.rad2deg(dtheta) + np.float32([80, 80, 50, 50, 0, 0])

def pixel_to_screen(x_img, y_img, h, w):
    ox, oy = w // 2, h // 2
    x_ndc = x_img / w
    y_ndc = 1 - (y_img / h)
    x_screen = (2 * x_ndc - 1)
    y_screen = (2 * y_ndc - 1)
    return x_screen, y_screen


arduino = serial.Serial(port="COM8", baudrate=115200, timeout=1)

lengths = list(i / 100 for i in [10, 12.4, 3.5])
z = 8

theta_series = np.deg2rad(np.linspace(10, 170, 4) - 90)
theta_array = np.float32([
    [t1, t2, t3, t4, 0, 0]
    for t1 in np.deg2rad(np.linspace(15, 165, 13) - 90)
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

radius_constraint1 = np.sqrt(reconstructed_points[..., 0]**2 + reconstructed_points[..., 1]**2) >= 0.1
radius_constraint2 = np.sqrt(reconstructed_points[..., 0]**2 + reconstructed_points[..., 1]**2) <= 0.2
mask = np.bitwise_and(radius_constraint1, radius_constraint2)

theta_array = theta_array[mask]
reconstructed_points = reconstructed_points[mask]
print(len(reconstructed_points))

plt.scatter(reconstructed_points[..., 0], reconstructed_points[..., 1])
plt.gca().set_aspect("equal")
plt.show()

data = {
    "raw_motor_0": [],
    "raw_motor_1": [],
    "raw_motor_2": [],
    "raw_motor_3": [],
    "motor_0": [], 
    "motor_1": [],
    "motor_2": [], 
    "motor_3": [],
    "x_expected": [],
    "y_expected": [],
    "z_expected": [],
    "x_img": [], 
    "y_img": [], 
    "x_screen": [], 
    "y_screen": []
}

cap = cv2.VideoCapture(0)
raw_angles = ikine(reconstructed_points[0], lengths)
angles = np.ceil(map_theta(raw_angles, np.deg2rad(np.float32([0, 90, 0, 0, 0, 0]))))
formatted = ",".join(f"{i}:{v}" for i, v in enumerate(angles))
byte_string = formatted.encode()
arduino.write(byte_string)
time.sleep(2)

df = None
print(len(reconstructed_points))

for pos in reconstructed_points:
    print(f"\nTesting position {pos}")
    raw_angles = ikine(pos, lengths)
    angles = np.ceil(map_theta(raw_angles, np.deg2rad(np.float32([0, 90, 0, 0, 0, 0]))))
    formatted = ",".join(f"{i}:{v}" for i, v in enumerate(angles))
    byte_string = formatted.encode()
    arduino.write(byte_string)
    time.sleep(4)

    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        detected = False  # Flag to check if any marker is detected

        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()

        # Detect ArUco markers
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is not None:
            print("Detected")
            detected = True
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == 0:
                    marker_corners = corners[i][0]
                    cX = np.mean(marker_corners[:, 0])
                    cY = np.mean(marker_corners[:, 1])
                    centroid_1 = (cX, cY)
                    print(f"Centroid of marker ID 0: {centroid_1}")
                    for i in range(4):
                        data[f"raw_motor_{i}"].append(raw_angles[i])
                        data[f"motor_{i}"].append(angles[i])
                    data["x_expected"].append(pos[0])
                    data["y_expected"].append(pos[1])
                    data["z_expected"].append(pos[2])
                    data["x_img"].append(cX)
                    data["y_img"].append(cY)
                    xs, ys = pixel_to_screen(cX, cY, h, w)
                    data["x_screen"].append(xs)
                    data["y_screen"].append(ys)
                    df = pd.DataFrame(data)
                    df.to_csv(f"robot_arm_cam{h}x{w}_real.csv", index=False)
                    break
        else:
            print("No markers detected.")

        if not detected:
            cv2.putText(frame, "No ArUco markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("ArUco Marker Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
