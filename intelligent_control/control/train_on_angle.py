import os

import neat
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
import multiprocessing
from utils.kinematics import fkine

def inv(p):
    x, y = p
    xndc = (x + 1) / 2
    yndc = (y + 1) / 2
    px = xndc * 640
    py = yndc * 480
    return [px, py]

# Carrega os dados do seu dataset
local_dir = os.path.dirname(__file__)
path = os.path.join(local_dir, 'extract_simulated_data/extracted_data.npz')
data = np.load(path)
positions_target = data['positions_target']
positions_motor = data['positions_motor']
#positions_inputs = np.delete(positions_inputs, 2, axis=1)  # Remove a coluna z
# angles_outputs = (data['joint_angles'][:, :4] / 180) - 0.5 # Normaliza os ângulos para [0, 1]
# angles_outputs = data['joint_angles'][:, :4] / 180
angles_target = data['joint_angles_target'][:, :4]
angles_motor = data['joint_angles_motor'][:, :4]
LINK_LENGTHS = [10.0, 12.4, 6.0]

# Avalia um genoma com base no erro médio
def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    total_angle_error = 0.0
    num_samples = len(angles_motor)

    for p1, p2, current_angles, target_angles in zip(positions_motor, positions_target, angles_motor, angles_target):
        pos_diff = p2 - p1  # Entrada da rede
        input_vec = pos_diff.tolist()

        # Rede prevê o delta do ângulo normalizado [0, 1]
        output = net.activate(input_vec)
        delta_angles = np.array(output) * 180.0  # Desnormaliza pra graus

        predicted_angles = current_angles + delta_angles
        predicted_angles /= 180.0
        target_angles /= 180.0
        angle_error = np.mean(np.abs(predicted_angles - target_angles))
        total_angle_error += angle_error

    avg_angle_error = total_angle_error / num_samples
    fitness = 1.0 / (1.0 + avg_angle_error)  # quanto menor o erro angular, maior a fitness

    return fitness

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

    winner = population.run(pe.evaluate, 50)  # Número de gerações
    print("\nMelhor indivíduo (fitness):", winner.fitness)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # salvar o próprio genoma (preferido, pois permite reconstruir a rede)
    return winner, winner_net

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    i = 1
    best_score = 0
    winner = 0
    winner_net = 0
    winner, winner_net =  run_neat(config_path)
    while(i < 1 and best_score < 0.98):
       print(f'ITERATION: {i}')
       current_winner, current_winner_net =  run_neat(config_path)
       if current_winner.fitness > best_score:
           winner = current_winner
           winner_net = current_winner_net
       best_score = winner.fitness
       i += 1

    with open('winner_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    for p1, p2, a1, a2 in zip(positions_motor, positions_target, angles_motor, angles_target):
        pos_diff = p2 - p1  # Entrada da rede
        input_vec = pos_diff.tolist()
        output = winner_net.activate(input_vec)
        delta_angles = np.array(output) * 180.0  # Desnormaliza pra graus

        predicted_angles = a1 + delta_angles

        # positions =  fkine(np.array([np.pi/2, 0, 0, 0]), [10.0, 12.4, 6.0], 10.0)
	    # print("Positions:\n", positions)

        
        print("target {!r}, target in cm {!r}, target in angle {!r}, got cm: {!r} and angle: {!r}".format(p2 , fkine((a2 / 180.0) * np.pi, LINK_LENGTHS), a2, fkine((predicted_angles / 180.0) * np.pi, LINK_LENGTHS), predicted_angles))
        #print("input {!r}, got {!r}".format(xi) , fkine(output, [10.0, 12.4, 6.0])))
    print("Winner genome:\n", winner.fitness)
