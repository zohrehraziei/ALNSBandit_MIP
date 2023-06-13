import numpy as np


class MIPConfig:
    num_destroy = 2
    num_repair = 1
    op_coupling = np.ones((2, 1))
    scores = [5, 3, 2, 1]  # Scores for new global best, better than current, accepted, and rejected
    mab_algorithm = 'epsilon-greedy'  # MAB algorithm used
    epsilon = 0.15
    max_iteration = 100  # Max number of iteration of ALNS
    max_no_improvement = 20  # Max number of iteration without improvement
    max_time = 100  # Max runtime in seconds
