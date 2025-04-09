import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("robot_arm_cam640x480.csv")

m1 = df["world_z"] >= 0.08
m2 = df["world_z"] <= 0.13
m3 = np.bitwise_and((df["world_p"].abs() < 0.1), (df["world_q"].abs() < 0.1))
m4 = np.bitwise_and(m1, m2)
mask = np.bitwise_and(m3, m4)

df_filtered = df[mask]

def extract_motors(row):
    pairs = row.split(',')
    motor_values = {f"motor_{i}": float(value) for i, value in 
                    [p.split(':') for p in pairs if int(p.split(':')[0]) < 4]}
    return pd.Series(motor_values)

motor_df = df_filtered['motor_control'].apply(extract_motors)
df_filtered = pd.concat([df_filtered, motor_df], axis=1)

lengths = list(i / 100 for i in [10, 12.4, 6, 3])
for i, l in enumerate(lengths):
    df_filtered[f'link_{i}'] = l

df_filtered.info()

print(df_filtered.iloc[np.random.randint(0, 30, 5)])