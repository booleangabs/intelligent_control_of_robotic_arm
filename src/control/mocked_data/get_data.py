from utils.kinematics import *

import numpy as np

def get_data():
	"""Get data from the robot arm and return the positions and joint angles."""
	# Define the lengths of the robot arm segments
	lengths = [10.0, 12.4, 6.0]

	# Define the fixed height of the actuator
	z = 10.0

	# Define the number of samples to generate
	num_samples = 500

	# Generate random x and y coordinates within a certain range
	x_coordinates = np.random.uniform(low=-25.0, high=25.0, size=num_samples)
	y_coordinates = np.random.uniform(low=0.0, high=25.0, size=num_samples)

	# Calculate the z coordinate based on the fixed height
	z_coordinates = np.full(num_samples, z)

	# Combine the x, y, and z coordinates into a single array
	positions = np.column_stack((x_coordinates, y_coordinates, z_coordinates))

	# Calculate the inverse kinematics for each position
	joint_angles = []
	for position in positions:
		joint_angle = ikine(position, lengths)
		joint_angles.append(joint_angle)

	# Return the positions
	return positions, joint_angles

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
