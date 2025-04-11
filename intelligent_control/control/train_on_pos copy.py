def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    total_position_error = 0.0
    num_samples = len(angles_motor)
    for p1, p2, current_angles, target_angles in zip(positions_motor, positions_target, angles_motor, angles_target):
        p1 = np.array(inv(p1))
        p2 = np.array(inv(p2))
        pos_diff = p2 - p1  # Entrada da rede
        input_vec = pos_diff.tolist()

        # Rede prevê o delta do ângulo normalizado [0, 1]
        output = net.activate(input_vec)
        delta_angles = (np.array(output) * np.pi) - np.pi/2 # Desnormaliza pra graus
        current_angles = ((current_angles / 180.0) * np.pi) - np.pi/2
        predicted_angles = current_angles + delta_angles
        predicted_position = identificador.prever_coordenadas(predicted_angles)
        # predicted_angles /= 180.0
        # target_angles /= 180.0
        position_error = np.mean(np.abs(predicted_position - p2))
        total_position_error += position_error

    avg_position_error = total_position_error / num_samples
    fitness = 1.0 / (1.0 + avg_position_error)  # quanto menor o erro angular, maior a fitness

    return fitnessdef evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    total_position_error = 0.0
    num_samples = len(angles_motor)
    for p1, p2, current_angles, target_angles in zip(positions_motor, positions_target, angles_motor, angles_target):
        p1 = np.array(inv(p1))
        p2 = np.array(inv(p2))
        pos_diff = p2 - p1  # Entrada da rede
        input_vec = pos_diff.tolist()

        # Rede prevê o delta do ângulo normalizado [0, 1]
        output = net.activate(input_vec)
        delta_angles = (np.array(output) * np.pi) - np.pi/2 # Desnormaliza pra graus
        current_angles = ((current_angles / 180.0) * np.pi) - np.pi/2
        predicted_angles = current_angles + delta_angles
        predicted_position = identificador.prever_coordenadas(predicted_angles)
        # predicted_angles /= 180.0
        # target_angles /= 180.0
        position_error = np.mean(np.abs(predicted_position - p2))
        total_position_error += position_error

    avg_position_error = total_position_error / num_samples
    fitness = 1.0 / (1.0 + avg_position_error)  # quanto menor o erro angular, maior a fitness

    return fitnessimport os

import neat
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
import multiprocessing
from utils.kinematics import fkine
from identification.identificacao import BracoIdentificacao

def inv(p):
    x, y = p
    xndc = (x + 1) / 2
    yndc = (y + 1) / 2
    px = xndc * 640
    py = yndc * 480
    return [px, py]

def ikine(position: np.ndarray, lenghts: list) -> np.ndarray:
    """Inverse kinematics for robotic arm

    Args:
        position (np.ndarray): XYZ position w.r.t. to base motor
        lenghts (list): First three stages lengths

    Returns:
        np.ndarray: Motor angles
    """
    x, y, z = position
    l1, l2, l3 = lenghts

    # phi = np.atan2(z, abs(y))  # Actual calculation
    phi = np.deg2rad(10)

    cphi, sphi = np.cos(phi), np.sin(phi)

    t1 = np.arctan2(y, x)

    new_x = np.sqrt(x**2 + y**2)
    x2 = new_x - l3 * abs(cphi)
    z2 = z - l3 * sphi

    c3 = (x2**2 + z2**2 - l1**2 - l2**2) / (2 * l1 * l2)

    t3 = -np.arccos(np.clip(c3, -1, 1))
    s3 = np.sin(t3)

    k1 = l1 + l2 * c3
    k2 = l2 * s3
    k3 = x2**2 + z2**2
    c2 = (k1 * x2 + k2 * z2) / k3
    s2 = (k1 * z2 - k2 * x2) / k3

    t2 = np.arctan2(s2, c2)
    t4 = phi - (t2 + t3)

    theta = np.float32([t1, t2, t3, t4, 0, 0])
    return theta

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

identificador = BracoIdentificacao(
    model_path='identification/modelos/modelo_cinematica_direta.pkl',
    scaler_X_path='identification/modelos/scaler_motores.pkl',
    scaler_y_path='identification/modelos/scaler_coordenadas.pkl'
)

# Avalia um genoma com base no erro médio
def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    total_position_error = 0.0
    num_samples = len(angles_motor)
    for p1, p2, current_angles, target_angles in zip(positions_motor, positions_target, angles_motor, angles_target):
        p1 = np.array(inv(p1))
        p2 = np.array(inv(p2))
        pos_diff = p2 - p1  # Entrada da rede
        pos_diff[0] /= 640
        pos_diff[1] /= 480
        input_vec = pos_diff.tolist()

        # Rede prevê o delta do ângulo normalizado [0, 1]
        output = net.activate(input_vec)
        max_delta = 0.05
        delta_pos = np.array(output) * max_delta #- np.pi/2 

        current_angles = ((current_angles / 180.0) * np.pi) - np.pi/2
        current_pos = fkine(current_angles, LINK_LENGTHS)
        current_pos[:2] = current_pos[:2] + delta_pos
        predicted_angles = ikine(current_pos, LINK_LENGTHS)
        predicted_position = identificador.prever_coordenadas(predicted_angles[:4])
        # predicted_angles /= 180.0
        # target_angles /= 180.0
        position_error = np.mean(np.abs(predicted_position - p2))
        total_position_error += position_error

    avg_position_error = total_position_error / num_samples
    fitness = 1.0 / (1.0 + avg_position_error)  # quanto menor o erro angular, maior a fitness

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
        p1 = np.array(inv(p1))
        p2 = np.array(inv(p2))
        pos_diff = p2 - p1  # Entrada da rede
        pos_diff[0] /= 640
        pos_diff[1] /= 480
        input_vec = pos_diff.tolist()
        output = winner_net.activate(input_vec)
        max_delta = 0.05
        delta_pos = np.array(output) * max_delta #- np.pi/2 

        current_angles = ((a1 / 180.0) * np.pi) - np.pi/2
        current_pos = fkine(current_angles, LINK_LENGTHS)
        current_pos[:2] = current_pos[:2] + delta_pos
        predicted_angles = ikine(current_pos, LINK_LENGTHS)
        predicted_position = identificador.prever_coordenadas(predicted_angles[:4])

        # positions =  fkine(np.array([np.pi/2, 0, 0, 0]), [10.0, 12.4, 6.0], 10.0)
	    # print("Positions:\n", positions)

        
        print("target {!r}, target in cm {!r}, target in angle {!r}, got cm: {!r} and angle: {!r} delta is: {!r}".format(p2 , fkine((a2 / 180.0) * np.pi, LINK_LENGTHS), a2, fkine((predicted_angles / 180.0) * np.pi, LINK_LENGTHS), predicted_angles,delta_pos))
        #print("input {!r}, got {!r}".format(xi) , fkine(output, [10.0, 12.4, 6.0])))
    print("Winner genome:\n", winner.fitness)
