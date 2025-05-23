import os
import neat
from utils.kinematics import *
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
import multiprocessing

# Carrega os dados do seu dataset
local_dir = os.path.dirname(__file__)
path = os.path.join(local_dir, 'mocked_data/mocked_data.npz')
data = np.load(path)
positions_inputs = data['positions']
#positions_inputs = np.delete(positions_inputs, 2, axis=1)  # Remove a coluna z
angles_outputs = data['joint_angles'][:, :4] / (2 * np.pi)  # Normaliza os ângulos para [0, 1]

lengths = [10.0, 12.4, 6.0]
z = 10.0

def angles_to_position(angles):
    """Convert joint angles to end-effector position."""
    # Convert angles to radians
    angles = np.array(angles)
    angles = angles * 2 * np.pi
    # Calculate the end-effector position using forward kinematics
    positions = fkine(angles, lengths, z)
    return positions

# Avalia um genoma com base no erro médio
def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    predictions = []
    targets = []

    for input_vec, target_vec in zip(positions_inputs, angles_outputs):
        input_vec[:2] = input_vec[:2]/ 25 # Normaliza os inputs para [-1, 1]
        output = net.activate(input_vec)
        positions_output = angles_to_position(output)  
        distance = np.linalg.norm(positions_output - input_vec)
        if distance > 10:
            return -1000
        else:
            predictions.append(output)            

        targets.append(target_vec)

    predictions = np.array(predictions)
    targets = np.array(targets)

    mse = mean_squared_error(targets, predictions)
    return 1.0 / (1.0 + mse)  # Quanto menor o erro, maior a fitness

# Avalia todos os genomas da população
def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness = evaluate_genome(genome, config)

# Roda o NEAT com a config fornecida
def run_neat(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), evaluate_genome)

    winner = population.run(pe.evaluate, 500)  # Número de gerações
    # print("\nMelhor indivíduo (fitness):", winner.fitness)

    # Salva a rede vencedora (opcional)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # Ou: salvar o próprio genoma (preferido, pois permite reconstruir a rede)
    return winner, winner_net

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    winner, winner_net =  run_neat(config_path)
    # i = 0
    # best_score = 0
    # winner = 0
    # winner_net = 0
    # while(i < 300 and best_score < 0.97):
    #    winner, winner_net =  run_neat(config_path)
    #    i += 1
    #    best_score = winner.fitness

    # with open('winner_genome.pkl', 'wb') as f:
    #     pickle.dump(winner, f)
    for xi, xo in zip(positions_inputs, angles_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r} pos{!r}".format(xi, xo, output, angles_to_position(output)))
    print("Winner genome:\n", winner.fitness)
