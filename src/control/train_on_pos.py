import os

import neat
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
import multiprocessing
import matplotlib.pyplot as plt
from utils.kinematics import fkine

# Carrega os dados do seu dataset
local_dir = os.path.dirname(__file__)
path = os.path.join(local_dir, 'extract_simulated_data/extracted_data.npz')
data = np.load(path)
positions_inputs = data['positions']
#positions_inputs = np.delete(positions_inputs, 2, axis=1)  # Remove a coluna z
angles_outputs = data['joint_angles'][:, :4] / (180)  # Normaliza os ângulos para [0, 1]

# Configurações da garra
LINK_LENGTHS = [10.0, 12.4, 6.0]
Z_ALTURA_FIXA = 10.0

# Avalia um genoma com base no erro da posição
def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    erros = []

    for entrada, saida_esperada in zip(positions_inputs, angles_outputs):
        saida_normalizada = net.activate(entrada)
        angulos_preditos = np.array(saida_normalizada) * 2 * np.pi  # Desnormaliza

        pos_predita = fkine(angulos_preditos, LINK_LENGTHS)
        erro_pos = np.linalg.norm(pos_predita[:2] - entrada)
        erros.append(erro_pos)

    erro_medio = np.mean(erros)
    return  1.0 / (1.0 + erro_medio)  # Fitness inversamente proporcional ao erro

# Avaliação de toda a população
def eval_genomes(genomes, config):
    for _, genome in genomes:
        evaluate_genome(genome, config)

# Gráfico de evolução da fitness
def plot_fitness(history):
    plt.plot(history)
    plt.xlabel("Geração")
    plt.ylabel("Fitness (1 / (1 + erro médio))")
    plt.title("Evolução da Fitness")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fitness_plot.png")
    plt.show()

# Executa o NEAT
def run_neat(config_path, num_generations=100):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), evaluate_genome)

    winner = population.run(pe.evaluate, num_generations)

    print("\nMelhor indivíduo (fitness):", winner.fitness)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    np.savez('winner_network.npz', weights=winner_net)

    # Gera gráfico
    fitness_history = [c.fitness for c in stats.most_fit_genomes]
    plot_fitness(fitness_history)
    return winner, winner_net

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    winner, winner_net =  run_neat(config_path)
    i = 0
    best_score = 0
    winner = 0
    winner_net = 0
    while(i < 5 and best_score < 0.965):
       print(f'ITERATION: {i}')
       winner, winner_net =  run_neat(config_path)
       i += 1
       best_score = winner.fitness

    with open('winner_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)
    for xi, xo in zip(positions_inputs, angles_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r} pos {!r}".format(xi, xo, output, fkine(output,LINK_LENGTHS)))
    print("Winner genome:\n", winner.fitness)
