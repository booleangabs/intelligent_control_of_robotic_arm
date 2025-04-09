from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import cv2
import numpy as np
import time
import traceback
import pandas as pd
from tqdm import tqdm


def pixel_to_screen(x_img, y_img, h, w):
    ox, oy = w // 2, h // 2
    x_ndc = x_img / w
    y_ndc = 1 - (y_img / h)
    x_screen = (2 * x_ndc - 1)
    y_screen = (2 * y_ndc - 1)
    return x_screen, y_screen

if __name__ == "__main__":
    client = RemoteAPIClient()
    sim = client.require('sim')

    # Initialize simulation
    sim.clearStringSignal("motorCtrl")
    sim.setStepping(True)

    sim.startSimulation()
    webcam_handle = sim.getObject("./webcam")
    # cv2.namedWindow("WIN")
    time.sleep(2)

    # Arm segment lengths
    lengths = [i / 100 for i in [10, 12.4, 6]]  # meters

    # Generate theta combinations
    theta_series = np.linspace(0, np.deg2rad(180), 7)
    theta_comb = np.float32([
        [t1, t2, t3, t4, 0, 0]
        for t1 in theta_series
        for t2 in theta_series
        for t3 in theta_series
        for t4 in theta_series
    ])
    theta_comb = np.clip(theta_comb, np.deg2rad(10), np.deg2rad(170))

    # Timing
    settling_delay = 2
    i = 0
    last_command_time = -settling_delay  # So first iteration runs immediately
    data = {
        "motor_control": [], 
        "world_x": [],
        "world_y": [], 
        "world_z": [],
        "world_p": [],
        "world_q": [], 
        "world_r": [],
        "x_img": [], 
        "y_img": [], 
        "x_screen": [], 
        "y_screen": []
    }
    try:
        pbar = tqdm(total = len(theta_comb))
        while (t := sim.getSimulationTime()) < len(theta_comb) * settling_delay * 2:
            if t - last_command_time >= settling_delay:
                if i == len(theta_comb):
                    break
                theta = theta_comb[i]
                i += 1
                pbar.update(1)

                # Send motor command
                motor_control = ",".join(f"{j}:{np.ceil(np.rad2deg(theta[j]))}" for j in range(6))
                sim.setStringSignal("motorCtrl", motor_control)
                last_command_time = t
                print(f"\n[INFO] Sent motor control: {motor_control} at sim time {t:.2f}s")
                continue  # Allow next loop(s) to run for 4.5s to settle

            # Camera capture
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
            command = None
            if (t - last_command_time) >= settling_delay - 0.05:
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 100:
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            centroid = (cx, cy)
                            contour = cnt
                            command = motor_control
                            break  # Only take first valid detection

            # Annotate & show image
            if centroid:
                cx, cy = centroid
                # Plane is the magenta marker in our simulation
                # I do know this is kind of cheating but we might not need to use this for anything
                # other than filtering points by "z"
                world_position = sim.getObjectPosition(sim.getObject(":/Plane"))
                world_orientation = sim.getObjectOrientation(sim.getObject(":/Plane"))
                x_screen, y_screen = pixel_to_screen(cx, cy, h=img.shape[0], w=img.shape[1])

                data["motor_control"].append(command)
                data["x_img"].append(cx)
                data["y_img"].append(cy)
                data["x_screen"].append(x_screen)
                data["y_screen"].append(y_screen)
                data["world_x"].append(world_position[0])
                data["world_y"].append(world_position[1])
                data["world_z"].append(world_position[2])
                data["world_p"].append(world_orientation[0])
                data["world_q"].append(world_orientation[1])
                data["world_r"].append(world_orientation[2])
                df = pd.DataFrame(data)
                df.to_csv(f"robot_arm_cam{img.shape[1]}x{img.shape[0]}.csv", index=False)
                print(f"[DATA LOGGED] Img: ({cx}, {cy})  Screen: ({x_screen:.3f}, {y_screen:.3f})  World: {world_position}")
                # cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
                # cv2.circle(img, centroid, 3, (100, 100, 100), -1)
                # cv2.putText(img, f"Centroid: ({centroid})", (10, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # cv2.putText(img, f"Motors: {motor_control}", (10, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            # cv2.imshow("WIN", img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            sim.step()

    except Exception as e:
        traceback.print_exc()

    finally:
        sim.stopSimulation()
        # cv2.destroyAllWindows()
        df = pd.DataFrame(data)
        df.to_csv(f"robot_arm_cam{img.shape[1]}x{img.shape[0]}.csv", index=False)
        print(f"[INFO] Saved data to robot_arm_cam{img.shape[1]}x{img.shape[0]}.csv")
