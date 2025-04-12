import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("robot_arm_cam480x640_real.csv")
df2 = pd.read_csv("robot_arm_cam480x640_real_10s.csv")


plt.figure()
plt.title("Coordenadas no campo de tela capturadas")
plt.scatter(df1["x_screen"], df1["y_screen"], label="Dados de treino e teste")
plt.scatter(df2["x_screen"], df2["y_screen"], label="Exemplo de dados a serem interpolados")
plt.gca().set_aspect("equal")
plt.legend()
plt.show()