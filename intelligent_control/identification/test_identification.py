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

df = pd.read_csv("robot_arm_cam480x640_real_10s.csv")

ang_cols = list([f"motor_{i}" for i in range(4)])
y_cols = ["x_img", "y_img"]

pred = identificador.prever_coordenadas(df[ang_cols])

plt.scatter(pred[..., 0], pred[..., 1])
plt.scatter(df["x_img"], df["y_img"])
plt.show()