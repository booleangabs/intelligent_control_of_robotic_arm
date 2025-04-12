# 🤖 Projeto de Identificação Cinemática de Braço Robótico

**Identificação da cinemática direta usando MLP**  
*IF705 - Automação Inteligente - CIn/UFPE*

---

## 📋 Instruções para a Equipe de Controle

### 1. **Pré-requisitos**
- Python 3.8+ instalado.
- Bibliotecas listadas no `requirements.txt`.

### 2. **Instalação**
```bash
git clone https://github.com/booleangabs/intelligent_control_of_robotic_arm
cd intelligent_control_of_robotic_arm/identificacao
pip install -r requirements.txt
```

### 3. **Arquivos Necessários**
Certifique-se de ter na pasta `modelos`:
- `modelo_cinematica_direta.pkl` → Modelo treinado.
- `scaler_motores.pkl` → Normalizador dos ângulos dos motores.
- `scaler_coordenadas.pkl` → Normalizador das coordenadas da câmera.

### 4. **Como Usar o Modelo no Controle**
#### Importe a classe `BracoIdentificacao`:
```python
from identificacao import BracoIdentificacao
```

#### Inicialize o identificador:
```python
identificador = BracoIdentificacao(
    model_path='modelos/modelo_cinematica_direta.pkl',
    scaler_X_path='modelos/scaler_motores.pkl',
    scaler_y_path='modelos/scaler_coordenadas.pkl'
)
```

#### Faça previsões de coordenadas:
```python
# Ângulos dos motores: [motor_0, motor_1, motor_2, motor_3]
angulos = [0.5, -0.3, 1.2, 0.8]
coordenadas = identificador.prever_coordenadas(angulos)
print(f"Posição prevista: x={coordenadas[0]:.1f}, y={coordenadas[1]:.1f}")
```

#### Saída esperada:
```
Posição prevista: x=320.5, y=240.3
```

### 5. **Integração com Arduino**
Use as coordenadas previstas para calcular erros e ajustar os ângulos via PWM:
```python
def enviar_comando_arduino(angulos):
    import serial
    arduino = serial.Serial('/dev/ttyUSB0', 115200)  # Porta do Arduino
    command = f"0:{angulos[0]},1:{angulos[1]},2:{angulos[2]},3:{angulos[3]}"
    arduino.write(command.encode())
```

---

## 🗂 Estrutura do Projeto
```
projeto_braco/
├── modelos/                   # Modelos e scalers
│   ├── modelo_cinematica_direta.pkl
│   ├── scaler_motores.pkl
│   └── scaler_coordenadas.pkl
├── identificacao.py           # Classe de identificação
├── exemplo_uso.py             # Exemplo para a equipe de controle
├── robot_arm_cam640x480.csv   # Dataset
└── requirements.txt           # Dependências
```

---

## 🚨 Troubleshooting
| Erro                          | Solução                      |
|-------------------------------|------------------------------|
| `FileNotFoundError`           | Verifique os caminhos dos arquivos `.pkl`. |
| `AttributeError`              | Certifique-se de que `carregar_modelo()` foi chamado. |
| `ValueError: Input contains NaN` | Valores de entrada fora da faixa de treinamento. |

---

## 📝 Como Contribuir
1. Faça um fork do repositório.
2. Crie uma branch: `git checkout -b minha-feature`.
3. Commit suas mudanças: `git commit -m 'Adicionei X'`.
4. Push para a branch: `git push origin minha-feature`.
5. Abra um Pull Request.

---

## 📧 Contato
**Equipe de Identificação**  
- João Pedro: `joao@cin.ufpe.br`  
- Colaborador: `colab@cin.ufpe.br`

---

🔧 **Manutenção**: Atualizado em 17/03/2024 | [Licença MIT](LICENSE)