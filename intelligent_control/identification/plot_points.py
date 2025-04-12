import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("robot_arm_cam480x640_real.csv")
print(len(df))

plt.figure()
plt.title("Image")
plt.scatter(df["x_img"], df["y_img"])
plt.gca().set_aspect("equal")

plt.figure()
plt.title("Screen")
plt.scatter(df["x_screen"], df["y_screen"])
plt.gca().set_aspect("equal")

plt.figure()
plt.title("World")
plt.scatter(df["x_expected"], df["y_expected"])
plt.gca().set_aspect("equal")

plt.show()