import cv2
import cv2.aruco as aruco
import numpy as np

# List of all available ArUco dictionaries
ARUCO_DICTS = [
    aruco.DICT_4X4_50,# aruco.DICT_4X4_100, aruco.DICT_4X4_250, aruco.DICT_4X4_1000,
    #aruco.DICT_5X5_50, aruco.DICT_5X5_100, aruco.DICT_5X5_250, aruco.DICT_5X5_1000,
    #aruco.DICT_6X6_50, aruco.DICT_6X6_100, aruco.DICT_6X6_250, aruco.DICT_6X6_1000,
    #aruco.DICT_7X7_50, aruco.DICT_7X7_100, aruco.DICT_7X7_250, aruco.DICT_7X7_1000,
    #aruco.DICT_ARUCO_ORIGINAL
]

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected = False  # Flag to check if any marker is detected

    for dict_id in ARUCO_DICTS:
        aruco_dict = aruco.getPredefinedDictionary(dict_id)
        parameters = aruco.DetectorParameters()

        # Detect ArUco markers
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            detected = True
            aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.putText(frame, f"Detected Dict: {dict_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
            break  # Stop checking other dictionaries once detected

    if not detected:
        cv2.putText(frame, "No ArUco markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("ArUco Marker Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
