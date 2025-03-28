import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import matplotlib.pyplot as plt

# Define the robot using Denavit-Hartenberg parameters
# Convert degrees to radians and centimeters to meters
deg2rad = np.pi / 180
cm2m = 0.01  # Convert centimeters to meters

# # Define the links using RevoluteDH (theta, d, a, alpha)
# # All alpha values are 0 since joints can be colinear (no twist)
# L1 = RevoluteDH(d=0.0, a=10 * cm2m, alpha=0 * deg2rad)        # Link 1: base to joint 1 (l_0 = 10 cm)
# L2 = RevoluteDH(d=0.0, a=10 * cm2m, alpha=0 * deg2rad)        # Link 2: joint 1 to joint 2 (l_1 = 10 cm)
# L3 = RevoluteDH(d=0.0, a=12.4 * cm2m, alpha=0 * deg2rad)      # Link 3: joint 2 to joint 3 (l_2 = 12.4 cm)
# L4 = RevoluteDH(d=0.0, a=6 * cm2m, alpha=0 * deg2rad)         # Link 4: joint 3 to joint 4 (l_3 = 6 cm)
# L5 = RevoluteDH(d=0.0, a=0 * cm2m, alpha=0 * deg2rad)         # Link 5: joint 4 to joint 5 (l_4 = 0 cm)
# L6 = RevoluteDH(d=0.0, a=3 * cm2m, alpha=0 * deg2rad)         # Link 6: joint 5 to joint 6 (l_5 = 3 cm)

# # Create the robot model
# robot = DHRobot([L1, L2, L3, L4, L5, L6], name="CustomRobotNoTwist")

# # Define the initial joint angles (in radians)
# theta_initial = np.array([0, 45, 45, 0, 0, 0]) * deg2rad

# # Print the robot structure
# print(robot)

# # Plot the robot in the initial configuration
# robot.plot(theta_initial, block=True)

# # Optional: Adjust the plot view for better visualization
# # robot.plot(theta_initial, backend='pyplot', block=True)

robot = rtb.models.Panda()

Te = robot.fkine(robot.qr)  # forward kinematics
print(Te)

Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
sol = robot.ik_LM(Tep)         # solve IK
print(sol)

q_pickup = sol[0]
print(robot.fkine(q_pickup)) 

qt = rtb.jtraj(robot.qr, q_pickup, 50)
robot.plot(qt.q, backend='pyplot', movie='panda1.gif')