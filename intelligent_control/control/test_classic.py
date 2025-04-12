import matplotlib.pyplot as plt
import pandas as pd
import serial
import time
import numpy as np
import cv2
import cv2.aruco as aruco

def fkine(theta: np.ndarray, lenghts: list, z: float) -> np.ndarray:
    t1, t2, t3, t4, *_ = theta
    l1, l2, l3 = lenghts

    d = (l1 * np.cos(t2) + l2 * np.cos(t2 + t3)) + l3 * np.cos(t2 + t3 + t4)
    x = np.cos(t1) * d
    y = np.sin(t1) * d
    return np.float32([x, y, z])

def ikine(position: np.ndarray, lenghts: list) -> np.ndarray:
    x, y, z = position
    l1, l2, l3 = lenghts
    phi = np.deg2rad(10)

    t1 = np.arctan2(y, x)
    new_x = np.sqrt(x**2 + y**2)
    x2 = new_x - l3 * abs(np.cos(phi))
    z2 = z - l3 * np.sin(phi)

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

    return np.float32([t1, t2, t3, t4, 0, 0])

def map_theta(target_theta, current_theta, directions=np.float32([1, 1, -1, 1, 1, 1])):
    dtheta = directions * (target_theta - current_theta)
    return np.rad2deg(dtheta) + np.float32([80, 80, 0, 50, 0, 0])

def pixel_to_screen(x_img, y_img, h, w):
    x_ndc = x_img / w
    y_ndc = 1 - (y_img / h)
    x_screen = (2 * x_ndc - 1)
    y_screen = (2 * y_ndc - 1)
    return x_screen, y_screen

# Serial connection
arduino = serial.Serial(port="COM7", baudrate=115200, timeout=1)
time.sleep(2)

lengths = [0.1, 0.124, 0.035]  # in meters
z = (10 + 2) / 100  # fixed Z in meters

initial_theta = np.deg2rad(np.float32([0, 90, 0, 0, 0, 0]))
raw_angles = ikine(np.float32([0.08, 0.0, z / 2]), lengths)
angles = np.ceil(map_theta(raw_angles, initial_theta))
formatted = ",".join(f"{i}:{v}" for i, v in enumerate(angles))
arduino.write(formatted.encode())
print("Sent signal")
time.sleep(2)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        marker_positions = {}

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in [0, 1]:
                print(marker_id)
                marker_corners = corners[i][0]
                cX = int(np.mean(marker_corners[:, 0]))
                cY = int(np.mean(marker_corners[:, 1]))
                marker_positions[marker_id] = (cX, cY)

                cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"ID {marker_id}", (cX + 10, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if 0 in marker_positions and 1 in marker_positions:
            Pg = marker_positions[0]
            Pt = marker_positions[1]

            Pg_screen = np.float32(pixel_to_screen(Pg[0], Pg[1], h, w))
            Pt_screen = np.float32(pixel_to_screen(Pt[0], Pt[1], h, w))

            e = Pg_screen - Pt_screen
           
            u_x = 0.02 * (e[0])
            u_y = 0.02 * (e[1])
            U = np.float32([u_x, u_y])
            
            Pn = fkine(raw_angles, lengths, z)
            Pn1 = Pn + np.float32([U[0], U[1], 0])

            theta_next = ikine(Pn1, lengths)
            angles_next = np.ceil(map_theta(theta_next, initial_theta))

            formatted = ",".join(f"{i}:{v}" for i, v in enumerate(angles_next))
            arduino.write(formatted.encode())
            time.sleep(3)

            raw_angles = theta_next 

    cv2.imshow("ArUco Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
