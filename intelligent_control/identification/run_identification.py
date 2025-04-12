import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class BracoIdentificacao:
    """Classe para identificação da cinemática direta do braço robótico."""
    
    def __init__(self, model_path=None, scaler_X_path=None, scaler_y_path=None):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
        if model_path and scaler_X_path and scaler_y_path:
            self.carregar_modelo(model_path, scaler_X_path, scaler_y_path)
    
    def treinar_modelo(self, csv_path='robot_arm_cam480x640_real.csv'):
        """Treina o modelo de cinemática direta a partir do CSV."""
        # Carregar dados
        df = pd.read_csv(csv_path)
        X = df[['motor_0', 'motor_1', 'motor_2', 'motor_3']]
        y = df[['x_img', 'y_img']]
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.15, 
            random_state=42,
            shuffle=True
        )
        
        # Normalização
        self.scaler_X = StandardScaler()
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        self.scaler_y = StandardScaler()
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # Treinar modelo
        self.model = MLPRegressor(
            hidden_layer_sizes=(256, 256, 256, 256),
            activation='relu',
            solver='adam',
            max_iter=2000,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train_scaled)
    
        # Avaliação
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MAE: {mae:.2f} pixels")
        print(f"R²: {r2:.4f}")
    
    def salvar_modelo(self, model_path, scaler_X_path, scaler_y_path):
        """Salva o modelo e scalers em arquivos."""
        # Garante que a pasta existe        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler_X, scaler_X_path)
        joblib.dump(self.scaler_y, scaler_y_path)
        print(f"Modelo salvo em: {model_path}")
    
    def carregar_modelo(self, model_path, scaler_X_path, scaler_y_path):
        """Carrega o modelo e scalers de arquivos."""
        try:
            self.model = joblib.load(model_path)
            self.scaler_X = joblib.load(scaler_X_path)
            self.scaler_y = joblib.load(scaler_y_path)
            print("Modelo carregado com sucesso!")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelo: {str(e)}")
    
    def prever_coordenadas(self, angulos_motores):
        """Prevê as coordenadas (x_img, y_img) a partir dos ângulos."""
        # Verifica se os componentes estão carregados
        if self.model is None or self.scaler_X is None or self.scaler_y is None:
            raise ValueError("Modelo ou scalers não foram carregados/carregados corretamente!")
        
        # Converte para DataFrame se necessário
        if not isinstance(angulos_motores, pd.DataFrame):
            angulos_df = pd.DataFrame(
                [angulos_motores],
                columns=['motor_0', 'motor_1', 'motor_2', 'motor_3']
            )
        else:
            angulos_df = angulos_motores
        
        # Faz a previsão
        X_scaled = self.scaler_X.transform(angulos_df)
        y_pred_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled)

if __name__ == "__main__":
    # Treinamento e salvamento automático
    identificador = BracoIdentificacao()
    identificador.treinar_modelo()
    
    # Caminhos relativos para a pasta 'modelos'
    identificador.salvar_modelo(
        model_path='modelo_cinematica_direta.pkl',
        scaler_X_path='scaler_motores.pkl',
        scaler_y_path='scaler_coordenadas.pkl'
    )