from identificacao import BracoIdentificacao
import numpy as np

# Exemplo de uso
identificador = BracoIdentificacao(
    model_path='modelos/modelo_cinematica_direta.pkl',
    scaler_X_path='modelos/scaler_motores.pkl',
    scaler_y_path='modelos/scaler_coordenadas.pkl'
)

angulos = [-1.56, -0.20, -0.34, 0.55]
coordenadas = identificador.prever_coordenadas(angulos)
print(f"Coordenadas previstas: x={coordenadas[0]:.1f}, y={coordenadas[1]:.1f}")