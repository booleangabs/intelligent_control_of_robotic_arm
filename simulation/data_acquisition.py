from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import os

if not os.path.isdir(os.path.join(os.getcwd(), "data")):
    os.mkdir(os.path.join(os.getcwd(), "data"))

client = RemoteAPIClient()
sim = client.require('sim')

sim.setStepping(True)

sim.startSimulation()
left_motor = sim.getObject("/Robot/left_motor")
right_motor = sim.getObject("/Robot/right_motor")

pwm_l_list = []
pwm_r_list = []

vel_l_list = []
vel_r_list = []

MAX_T = 30 # [s]
try:
    while (t := sim.getSimulationTime()) < MAX_T:
        pwm_l = min(t / MAX_T, 1)
        pwm_r =  min(t / MAX_T, 1)
        sim.setFloatSignal("left_motor_pwm", pwm_l)
        sim.setFloatSignal("right_motor_pwm", pwm_r)

        left_w = sim.getJointVelocity(left_motor)
        right_w = sim.getJointVelocity(left_motor)

        sleep(0.005)

        pwm_l_list.append(pwm_l)
        vel_l_list.append(left_w)

        pwm_r_list.append(pwm_r)
        vel_r_list.append(right_w)

        sim.step()
except Exception as e:
    print(e)
finally:
    sim.stopSimulation()
    np.save("data/pwm_l.npy", pwm_l_list)
    np.save("data/pwm_r.npy", pwm_r_list)
    np.save("data/vel_l.npy", vel_l_list)
    np.save("data/vel_r.npy", vel_r_list)
    fig, ax = plt.subplots(1, 2)
    plt.suptitle("PWM duty cycle vs Wheel speed [rad/s]")
    ax[0].scatter(pwm_l_list, vel_l_list)
    ax[0].title.set_text("left")
    ax[1].scatter(pwm_r_list, vel_r_list)
    ax[1].title.set_text("right")
    plt.show()