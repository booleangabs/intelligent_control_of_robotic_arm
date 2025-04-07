from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import cv2
import time
import numpy as np

client = RemoteAPIClient()
sim = client.require('sim')
clip_angle = lambda x: min(max(x, 10), 170)

sim.clearStringSignal("motorCtrl")
sim.setStepping(True)
sim.startSimulation()
motor_control = f"0:0,1:90,2:90,3:90"
sim.setStringSignal("motorCtrl", motor_control)

webcam_handle = sim.getObject("./webcam")
print(webcam_handle)
cv2.namedWindow("WIN")
time.sleep(2)
try:
    while (t := sim.getSimulationTime()) < 15:
        if t > 7:
            sim.setStringSignal("motorCtrl", "0:180")
        print(f'Simulation time: {t:.2f} [s]')
        img, res = sim.getVisionSensorImg(webcam_handle)
        img = np.reshape(sim.unpackUInt8Table(img), res[::-1] + [3])
        img = np.flipud(img)
        img = img[..., ::-1].astype("uint8")
        cv2.imshow("WIN", img)
        if cv2.waitKey(1):
            pass
        sim.step()
except Exception as e:
    print(e)
finally:
    sim.stopSimulation()
    cv2.destroyAllWindows()