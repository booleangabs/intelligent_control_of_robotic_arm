import numpy as np

pwm_l = np.load("data/pwm_l.npy")
pwm_r = np.load("data/pwm_r.npy")
vel_l = np.load("data/vel_l.npy")
vel_r = np.load("data/vel_r.npy")

max_left = np.mean(vel_l[pwm_l > 0.99])
max_right = np.mean(vel_l[pwm_l > 0.99])
print(f"Max speed: {(max_left + max_right) / 2:.5f} rad/s")

with open("data/max_speed.txt", "w") as file:
    file.write(f"{(max_left + max_right) / 2:.5f}")