from utils.kinematics import *

import numpy as np

def get_data():
    """Gera N posições aleatórias válidas para a garra robótica com ângulos válidos (sem NaN)."""
    # Define the lengths of the robot arm segments
    lengths = [10.0, 12.4, 6.0]
    
	# Define the fixed height of the actuator
    z_fixed = 10.0
    
	# Define sample size
    num_samples = 50

    positions = []
    joint_angles = []

    while len(positions) < num_samples:
        x = np.random.uniform(-25.0, 25.0)
        y = np.random.uniform(0.0, 25.0)
        z = z_fixed
        position = np.array([x, y, z])

        angles = ikine(position, lengths)

        if np.any(np.isnan(angles)):
            continue  # posição inválida, ignora

        positions.append(position)
        joint_angles.append(angles)

    return np.array(positions), np.array(joint_angles)

if __name__ == "__main__":
	positions, joint_angles = get_data()
	print("Positions:\n", positions)
	print("Joint Angles:\n", joint_angles)
	# Save the data to a file
	np.savez("control/mocked_data/mocked_data.npz", positions=positions, joint_angles=joint_angles)

	# positions =  fkine(np.array([np.pi/2, 0, 0, 0]), [10.0, 12.4, 6.0], 10.0)
	# print("Positions:\n", positions)

	# angles =  ikine(np.array([0,  28.4,  10. ]), [10.0, 12.4, 6.0])
	# print("Angles:\n", angles)
