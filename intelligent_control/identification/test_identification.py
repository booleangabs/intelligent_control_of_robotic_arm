from run_identification import BracoIdentificacao
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Exemplo de uso
identificador = BracoIdentificacao(
    model_path='modelo_cinematica_direta.pkl',
    scaler_X_path='scaler_motores.pkl',
    scaler_y_path='scaler_coordenadas.pkl'
)

df1 = pd.read_csv("robot_arm_cam480x640_real.csv")
df2 = pd.read_csv("robot_arm_cam480x640_real_10s.csv")


ang_cols = list([f"motor_{i}" for i in range(4)])
y_cols = ["x_img", "y_img"]

pred = identificador.prever_coordenadas(df1[ang_cols])

fig, ax = plt.subplots(1, 2)
ax[0].title.set_text(f"Performance nos dados de treino/teste.\nMean Abs. Error (MAE): {(df1[y_cols] - pred).abs().mean().mean():.4f} [pixels]")
ax[0].scatter(pred[..., 0], pred[..., 1])
ax[0].scatter(df1["x_img"], df1["y_img"])
ax[0].set_aspect("equal")

pred = identificador.prever_coordenadas(df2[ang_cols])

ax[1].title.set_text(f"Performance nos dados de interpolação.\nMean Abs. Error (MAE): {(df2[y_cols] - pred).abs().mean().mean():.4f} [pixels]")
ax[1].scatter(pred[..., 0], pred[..., 1])
ax[1].scatter(df2["x_img"], df2["y_img"])
ax[1].set_aspect("equal")

plt.show()