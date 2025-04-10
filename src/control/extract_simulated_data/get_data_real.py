import pandas as pd
import numpy as np

# Path to the CSV file
csv_file = 'control/extract_simulated_data/robot_arm_cam480x640_real.csv'

# Columns to select
selected_columns_angles_raw = ['raw_motor_0', 'raw_motor_1', 'raw_motor_2','raw_motor_3']
selected_columns_angles = ['motor_0', 'motor_1', 'motor_2','motor_3']
selected_columns_positions = ['x_screen', 'y_screen']

# Read the CSV file
data = pd.read_csv(csv_file)

# Extract the selected columns
positions = data[selected_columns_positions]
joint_angles = data[selected_columns_angles]

# Convert to numpy arrays
positions = positions.to_numpy()
joint_angles = joint_angles.to_numpy()

print('terminou')
np.savez("control/extract_simulated_data/extracted_data.npz", positions=positions, joint_angles=joint_angles)