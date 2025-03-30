import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH

deg2rad = np.pi / 180
cm2m = 0.01

L1 = RevoluteDH(d=10 * cm2m, a=0.0, alpha=np.pi / 2, offset=0)
L2 = RevoluteDH(d=0, a=12.4 * cm2m, alpha=0, offset=0)
L3 = RevoluteDH(d=0.0, a=0, alpha=-np.pi / 2, offset=np.pi) 
L4 = RevoluteDH(d=6 * cm2m, a=0, alpha=np.pi / 2, offset=0)
L5 = RevoluteDH(d=0.0, a=0 * cm2m, alpha=np.pi / 2, offset=np.pi / 2)
L6 = RevoluteDH(d=3 * cm2m, a=0, alpha=0, offset=0)
robot = DHRobot([L1, L2, L3, L4, L5, L6], name="Beru")

trajectory = np.linspace(0, np.pi, 100).reshape(-1, 1)
zeros = np.zeros((100, 1))
thetas = np.concatenate(
    [
        trajectory, 
        zeros + np.pi / 4,
        zeros + (3 * np.pi / 4), 
        zeros,
        zeros + np.pi / 2, 
        zeros
    ], 
    axis=1
)

print(robot)

robot.plot(thetas, block=True, backend="pyplot", loop = True) # loop = False