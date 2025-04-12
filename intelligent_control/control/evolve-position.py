"""
"""

import os

import neat
from utils.kinematics import *
import numpy as np
import multiprocessing

lengths = [10.0, 12.4, 6.0]
z = 10.0

data = np.load('control/mocked_data/mocked_data.npz')
positions_inputs = data['positions']
# positions_inputs = np.delete(positions_inputs, 2, axis=1)
angles_outputs = data['joint_angles']
angles_outputs = angles_outputs[:, :4]/(2*np.pi)  # Normalize angles to [0, 1] range

def angles_to_position(angles):
    """Convert joint angles to end-effector position."""
    # Convert angles to radians
    angles = np.array(angles)
    angles = angles * 2 * np.pi
    # Calculate the end-effector position using forward kinematics
    positions = fkine(angles, lengths, z)
    return positions

def angular_distance(theta_d, theta_g):
    return min(abs(theta_d - theta_g), 2 * np.pi - abs(theta_d - theta_g))

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    erro = []
    for xi in positions_inputs:
        output = net.activate(xi)
        positions_output = angles_to_position(output)
        distance = np.linalg.norm(positions_inputs - positions_output)
        if distance > 200:
            return -1000
        else:
            erro.append(distance)
    return  - np.max(erro)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    config.checkpoint_interval = 0
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.StdOutReporter(True))
    # p.add_reporter(neat.Checkpointer(5))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    
    # Run for up to 300 generations.
    winner = p.run(pe.evaluate, 1000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(positions_inputs, angles_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r} pos {!r}".format(xi, xo, output, angles_to_position(output)))

    # node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)


if __name__ == '__main__':
    print(angles_outputs)