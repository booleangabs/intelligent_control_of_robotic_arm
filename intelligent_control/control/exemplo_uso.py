import pickle
import numpy as np
import neat

# Supondo que X_test tem shape (500, 4)
pos = [0.5390625,-0.5520833333333333]

# Carrega o modelo salvo
with open('models/winner_genome_97_angle.pkl', 'rb') as f:
    genome = pickle.load(f)
    
# Carrega a configuração usada durante o treino
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'config-feedforward'  # esse é o arquivo de configuração
)

# Cria a rede neural a partir do genoma + config
net = neat.nn.FeedForwardNetwork.create(genome, config)
# Faz predição
angles = net.activate(pos)
angles = [s * 180 for s in angles]

print(angles) 

