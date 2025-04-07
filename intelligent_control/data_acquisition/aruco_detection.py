import cv2
import cv2.aruco as aruco
import numpy as np

# List of all available ArUco dictionaries
ARUCO_DICTS = [
    aruco.DICT_4X4_50
]

def img2real(pixel, camera_height):
    focal_length = 1500
    real_x = -(camera_height * pixel[0]) / focal_length
    real_y = (camera_height * pixel[1]) / focal_length
    return np.float32([real_x, real_y])

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
        print(frame.shape)
        if ids is not None:
            detected = True
            aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.putText(frame, f"Detected Dict: {dict_id} | {img2real(corners[0][0][0], 45)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
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
