# import random
import logging
import numpy as np
import numpy.random as rnd
import pyscipopt as scip
# import matplotlib.pyplot as plt

from State import StateSCIP, ContextualState
from alns.select import MABSelector
from alns.ALNS import ALNS
from alns.accept import HillClimbing
from alns.stop import MaxIterations
from mip_config import MIPConfig
from mabwiser.mab import LearningPolicy, NeighborhoodPolicy

logger = logging.getLogger(__name__)


def random_remove(state: StateSCIP, rnd_state):
    # Copy the model's variables
    vars = state.model.getVars()

    # Create a new SCIP solution
    sol = state.model.createSol()

    # Randomly select a fraction of variables set to 1 and unset them
    ones = [var for var in vars if state.model.getSolVal(sol, var) == 1 and var.getLbLocal() != var.getUbLocal()]
    to_remove = rnd_state.choice(ones, size=int(len(ones) * 0.2), replace=False)  # remove 20% of vars

    for var in to_remove:
        if var.getLbLocal() == 0:  # Check if variable cannot be set to zero
            continue
        state.model.setSolVal(sol, var, 0)

    return sol


def worse_remove(state: StateSCIP, rnd_state):
    # Copy the model's variables
    vars = state.model.getVars()

    # If there is no solution, remove anything
    if vars is None or len(vars) == 0:
        return state.model.createSol()

    # Create a new SCIP solution
    sol = state.model.createSol()

    # Identify variables that contribute least to the objective function
    ones = [(var, var.getObj()) for var in vars if var.getLbLocal() != var.getUbLocal()]
    ones.sort(key=lambda x: x[1])  # sort by coefficient
    to_remove = [var for var, _ in ones[:int(len(ones) * 0.2)]]  # remove 20% vars with small coefficient

    for var in to_remove:
        if var.getLbLocal() == 0:  # Check if variable cannot be set to zero
            continue
        state.model.setSolVal(sol, var, 0)

    return sol


def random_repair(state: StateSCIP, rnd_state):
    # Copy the model's variables
    vars = state.model.getVars()

    # If there is no solution, we cannot repair anything
    if vars is None or len(vars) == 0:
        return state.model.createSol()

    # Create a new SCIP solution
    sol = state.model.createSol()

    zeros = [var for var in vars if var.getLbLocal() != var.getUbLocal() and
             state.model.getSolVal(state.model.getSols(), var) == 0]

    if len(zeros) > 0:
        # Randomly choose a variable from the zeros list
        chosen_var = rnd_state.choice(zeros)

        # Set the chosen variable to 1 in the solution
        state.model.setSolVal(sol, chosen_var, 1.0)

    return sol


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

    # if model.getStatus() == "optimal":
    #     print("initial model optimized")

    # Record LP solution
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
    logger.info(f"Running BanditALNS for instance: {instance_path}")

    mip_instance = read_mps(instance_path)


    # Initialize ALNS with random state
    initial_sol = init_sol(model=mip_instance)

    alns = ALNS(random_state)

    alns.add_destroy_operator(random_remove)
    alns.add_destroy_operator(worse_remove)
    alns.add_repair_operator(random_repair)

    if initial_sol is None:
        logger.error("failed to initialize solution within time limit")
        return

    # StateSCIP.get_context = StateSCIP.get_mip_context
    ContextualState.get_context = StateSCIP.get_context
    print("get context assigned")


    op_select = MABSelector(
        scores=MIPConfig.scores,
        num_destroy=MIPConfig.num_destroy,
        num_repair=MIPConfig.num_repair,
        # learning_policy=LearningPolicy.LinGreedy(epsilon=0.15),
        learning_policy=LearningPolicy.LinUCB(alpha=1.25, l2_lambda=1),
        neighborhood_policy=None)

    stop = MaxIterations(1000)
    accept = HillClimbing()


    best_objective = float('inf')

    for i in range(stop.max_iterations):
        result = alns.iterate(initial_sol, op_select, accept, stop)
        print("alns is run")
        state = result.best_state

        if state.objective() < best_objective:
            best_objective = state.objective()
    print(f"Found solution with objective {best_objective}.")


if __name__ == "__main__":
    instance_path = "C:/Users/a739095/Streamfolder/New_Forked_ALNS_CMAB_MIP/ALNSBandit_MIP/data/" \
                    "noswot.mps.gz"
    # Create MIP instance
    run_banditalns(instance_path)
