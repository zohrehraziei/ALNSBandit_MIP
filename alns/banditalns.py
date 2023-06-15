# import random
import logging
import numpy as np
import numpy.random as rnd
import pyscipopt as scip
import matplotlib.pyplot as plt

from alns.ALNS import ALNS
from mip_config import MIPConfig
from alns.select import MABSelector
from alns.accept import HillClimbing
from alns.stop import MaxIterations
from State import StateSCIP
from mabwiser.mab import LearningPolicy, NeighborhoodPolicy

logger = logging.getLogger(__name__)


def random_remove(state: StateSCIP, rnd_state):
    # Copy the model's variables and create a new solution
    vars = state.model.getVars()
    sol = state.model.createSol()

    # Randomly select a fraction of variables set to 1 and unset them
    ones = [var for var in vars if state.model.getSolVal(sol, var) == 1 and var.getLbLocal() != var.getUbLocal()]
    to_remove = rnd_state.choice(ones, size=int(len(ones) * 0.2), replace=False)  # remove 20% of vars

    for var in to_remove:
        if var.getLbLocal() == 0:  # Check if variable cannot be set to zero
            continue
        state.model.setSolVal(sol, var, 0)

    return StateSCIP(state.model)


def worse_remove(state: StateSCIP, rnd_state) -> StateSCIP:
    # Copy the model's variables and create a new solution
    vars = state.model.getVars()
    sol = state.model.createSol()

    # If there is no solution, remove anything
    if vars is None or len(vars) == 0:
        return state

    # Identify variables that contribute least to the objective function
    ones = [(var, var.getObj()) for var in vars if state.model.getSolVal(sol, var) == 1 and var.getLbLocal() != var.getUbLocal()]
    ones.sort(key=lambda x: x[1])  # sort by coefficient
    to_remove = [var for var, _ in ones[:int(len(ones) * 0.2)]]  # remove 20% vars with small coefficient

    for var in to_remove:
        if var.getLbLocal() == 0:  # Check if variable cannot be set to zero
            continue
        state.model.setSolVal(sol, var, 0)

    return StateSCIP(state.model)


def random_repair(state: StateSCIP, rnd_state) -> StateSCIP:
    # Copy the model's variables and create a new solution
    vars = state.model.getVars()
    sol = state.model.createSol()

    # If there is no solution, we cannot repair anything
    if vars is None or len(vars) == 0:
        return state

    zeros = [var for var in vars if state.model.getSolVal(sol, var) == 0 and var.getLbLocal() != var.getUbLocal()]

    # Randomly choose a variable from the zeros list
    chosen_var = rnd_state.choice(zeros)

    # Set the chosen variable to 1 in the solution
    state.model.setSolVal(sol, chosen_var, 1)

    return StateSCIP(state.model)


def read_mps(file_path):
    """
    Read MIP instance from an MPS file.

    Parameters
    ----------
        file_path: Path to MPS file.

    Returns
    -------
        model: The SCIP model containing the MIP instance.
    """
    model = scip.Model()
    model.readProblem(file_path)

    return model


def init_sol(model):
    # Keep a record of the original variables and their types
    original_vars = model.getVars()
    original_vtypes = {var.name: var.vtype() for var in original_vars}

    # Solve LP relaxation
    for var in original_vars:
        model.chgVarType(var, 'CONTINUOUS')
    model.optimize()

    # Record LP solutio
    x_star = {var.name: model.getVal(var) for var in original_vars}

    # Round x_star to obtain x_tilda
    x_tilda = {var_name: round(value) for var_name, value in x_star.items()}

    # Free transformation and reinitialize problem
    model.freeTransform()
    for var in original_vars:
        model.chgVarType(var, original_vtypes[var.name])

    # Update model to use x_tilda as new lower bounds
    for var in original_vars:
        model.chgVarLb(var, x_tilda[var.name])

    # Solve model again
    model.optimize()

    return StateSCIP(model)


def run_banditalns(instance_path):
    SEED = 7654
    random_state = np.random.RandomState(SEED)
    alns = ALNS(random_state)

    alns.add_destroy_operator(random_remove)
    alns.add_destroy_operator(worse_remove)
    alns.add_repair_operator(random_repair)


    logger.info(f"Running BanditALNS for instance: {instance_path}")

    mip_instance = read_mps(instance_path)
    initial_state = StateSCIP(mip_instance)
    StateSCIP.get_context = StateSCIP.get_context


    # Initialize ALNS with random state
    initial_sol = init_sol(model=mip_instance)

    if initial_sol is None:
        logger.error("failed to initialize solution within time limit")
        return

    def get_context(state):
        return StateSCIP.get_context()

    op_select = MABSelector(
        scores=MIPConfig.scores,
        num_destroy=MIPConfig.num_destroy,
        num_repair=MIPConfig.num_repair,
        learning_policy=LearningPolicy.LinGreedy(epsilon=0.15),
        neighborhood_policy=None)

    stop = MaxIterations(50)
    accept = HillClimbing()

    res = alns.iterate(initial_sol, op_select, accept, stop)
    print(f"found solution with objective {res.best_state.objective()}.")

    _, ax = plt.subplots(figsize=(12, 6))
    res.plot_objectives(ax=ax, lw=2)


if __name__ == "__main__":
    instance_path = "C:/Users/a739095/Streamfolder/Forked_ALNS_CMAB_MIP/ALNS/data/gen-ip002.mps.gz"
    # Create MIP instance
    run_banditalns(instance_path)

# # We'll need to keep track of the best and current states
#     best_state = current_state = initial_sol
#     iterations = 0
#
#     # Here, we directly call stop as a callable in the while condition
#     while not stop(random_state, best_state, current_state):
#         # apply the operators to get a new candidate solution and score
#         candidate_state, destroy_idx, repair_idx, score = alns.iterate(current_state, op_select, accept,
#                                                                        stop)
#
#         # update MABSelector with the candidate solution, the operators used, and the outcome
#         outcome = MIPConfig.scores.index(score) if score in MIPConfig.scores else len(MIPConfig.scores)
#         op_select.update(candidate_state, destroy_idx, repair_idx, outcome)
#
#         # Update the current and best states
#         current_state = candidate_state
#         if current_state.score > best_state.score:
#             best_state = current_state
#
#         iterations += 1
