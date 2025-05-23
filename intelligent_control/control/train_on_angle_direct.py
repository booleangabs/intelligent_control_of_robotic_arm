import os

import neat
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
import multiprocessing

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
positions_inputs = data['positions_target']
#positions_inputs = np.delete(positions_inputs, 2, axis=1)  # Remove a coluna z
# angles_outputs = (data['joint_angles'][:, :4] / 180) - 0.5 # Normaliza os ângulos para [0, 1]
# angles_outputs = data['joint_angles'][:, :4] / 180
angles_outputs = data['joint_angles_target'][:, :4]  # Normaliza os ângulos para [-1, 1]

# Avalia um genoma com base no erro médio
def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    predictions = []
    targets = []

    for input_vec, target_vec in zip(positions_inputs, angles_outputs):
        # target_vec = (target_vec + np.pi/2) / (np.pi)  # Normaliza a as predictions para [0, 1]
        # target_vec = target_vec / np.pi/2 # Normaliza a as predictions para [-1, 1]

        target_vec = target_vec / 180  # Normaliza a as predictions para [0, 1]
        output = net.activate(input_vec)  
        predictions.append(output)
        targets.append(target_vec)

    predictions = np.array(predictions) 
    targets = np.array(targets) 

    # diff = np.abs((predictions - targets + 0.5) % 1.0 - 0.5)  # Diferença cíclica
    # ang_error = np.mean(diff)
    # return 1.0 / (1.0 + ang_error)


    # mae = np.mean(np.abs(targets - predictions))
    # return 1.0 / (1.0 + mae)

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

    winner = population.run(pe.evaluate, 100)  # Número de gerações
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
    while(i < 5 and best_score < 0.98):
       print(f'ITERATION: {i}')
       current_winner, current_winner_net =  run_neat(config_path)
       if current_winner.fitness > best_score:
           winner = current_winner
           winner_net = current_winner_net
       best_score = winner.fitness
       i += 1

    with open('winner_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    for xi, xo in zip(positions_inputs, angles_outputs):
        output = winner_net.activate(xi)

        # output = (np.array(output) + 0.5) * 180
        # xo = (xo + 0.5) * 180

        # output = np.array(output) * 180 
        # xo = xo * 180

        # Para converter para graus output de [-1, 1] para [0, 180]
        # output = (np.array(output) + 1) * 90
        # xo = (xo + np.pi/2) * (180/np.pi)

        # Para converter para graus output de [0, 1] para [0, 180]
        output = np.array(output) * 180 
        # xo = (xo + np.pi/2) * (180/np.pi)
        

        print("input {!r}, expected output {!r}, got {!r}".format(inv(xi) , xo, output))
        #print("input {!r}, got {!r}".format(xi) , fkine(output, [10.0, 12.4, 6.0])))
    print("Winner genome:\n", winner.fitness)
