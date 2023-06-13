# import random
import logging
import numpy as np
import numpy.random as rnd
import pyscipopt as scip

from alns.ALNS import ALNS
from mip_config import MIPConfig
from alns.select import MABSelector
from alns.accept import HillClimbing
from alns.stop import MaxIterations
from State import SCIPState
from alns.context_extractor import ContextExtractor
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

logger = logging.getLogger(__name__)


def random_remove(state, rnd_state):
    destroyed_state = state.copy()
    model = destroyed_state.get_model()
    vars = model.getVars()
    sol = model.createSol()

    # Randomly select a fraction of variables set to 1 and unset them
    ones = [var for var in vars if model.getSolVal(sol, var) == 1 and var.getLbLocal() != var.getUbLocal()]
    to_remove = rnd_state.choice(ones, size=int(len(ones) * 0.2), replace=False)  # remove 20% of vars

    for var in to_remove:
        model.setSolVal(sol, var, 0)

    return destroyed_state


def worse_remove(state, rnd_state):
    destroyed_state = state.copy()
    model = destroyed_state.get_model()
    vars = model.getVars()
    sol = model.createSol()

    # If there is no solution, remove anything
    if vars is None or len(vars) == 0:
        return state

    # Identify variables that contribute least to the objective function
    ones = [(var, var.getObj()) for var in vars if
            model.getSolVal(sol, var) == 1 and var.getLbLocal() != var.getUbLocal()]
    ones.sort(key=lambda x: x[1])  # sort by coefficient
    to_remove = [var for var, _ in ones[:int(len(ones) * 0.2)]]  # remove 20% vars with small coefficient

    for var in to_remove:
        model.setSolVal(sol, var, 0)

    return destroyed_state


def random_repair(state, rnd_state):
    repaired_state = state.copy()
    model = repaired_state.get_model()
    vars = model.getVars()
    sol = model.createSol()

    # If there is no solution, we cannot repair anything
    if vars is None or len(vars) == 0:
        return state

    zeros = [var for var in vars if model.getSolVal(sol, var) == 0 and var.getLbLocal() != var.getUbLocal()]

    # Randomly choose a variable from the zeros list
    chosen_var = rnd_state.choice(zeros)

    # Set the chosen variable to 1 in the solution
    model.setSolVal(sol, chosen_var, 1)

    return repaired_state


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
    # x_star_new = {var.name: model.getVal(var) for var in original_vars}

    return SCIPState(model)


def run_banditalns(instance_path):
    SEED = 7654
    random_state = np.random.RandomState(SEED)

    logger.info(f"Running BanditALNS for instance: {instance_path}")

    mip_instance = read_mps(instance_path)
    vars = mip_instance.getVars()
    # destroy_operators_1 = DestroyOperator(mip_instance)
    # destroy_operators_2 = DestroyOperator(mip_instance)
    # repair_operator = RepairOperator(mip_instance)

    # Initialize ALNS with random state
    initial_sol = init_sol(model=mip_instance)

    if initial_sol is None:
        logger.error("failed to initialize solution within time limit")
        return
    initial_state = SCIPState(instance_path)
    alns = ALNS(random_state)

    # alns.add_destroy_operator(lambda state, rnd_state: random_remove(state, rnd_state))
    alns.add_destroy_operator(random_remove)
    alns.add_destroy_operator(worse_remove)
    alns.add_repair_operator(random_repair)


    # Extract static features
    context_extractor = ContextExtractor(problem_instance_file=instance_path)
    static_contexts = context_extractor.extract_context(initial_state)

    arms = MABSelector.make_arms(num_destroy=MIPConfig.num_destroy, num_repair=MIPConfig.num_repair,
                                 op_coupling=MIPConfig.op_coupling)
    op_select = MABSelector(
        scores=MIPConfig.scores,
        mab=MAB(arms, learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15),
                neighborhood_policy=NeighborhoodPolicy.Radius(radius=1)),
        num_destroy=MIPConfig.num_destroy,
        num_repair=MIPConfig.num_repair,
        context_extractor=lambda state: static_contexts)

    stop = MaxIterations(50)
    accept = HillClimbing()

    # We'll need to keep track of the best and current states
    best_state = current_state = initial_sol
    iterations = 0

    # Here, we directly call stop as a callable in the while condition
    while not stop(random_state, best_state, current_state):
        # apply the operators to get a new candidate solution and score
        candidate_state, destroy_idx, repair_idx, score = alns.iterate(current_state, op_select, accept,
                                                                       stop)

        # update MABSelector with the candidate solution, the operators used, and the outcome
        outcome = MIPConfig.scores.index(score) if score in MIPConfig.scores else len(MIPConfig.scores)
        op_select.update(candidate_state, destroy_idx, repair_idx, outcome)

        # Update the current and best states
        current_state = candidate_state
        if current_state.score > best_state.score:
            best_state = current_state

        iterations += 1

    # Final solution
    print(best_state)


if __name__ == "__main__":
    instance_path = "C:/Users/a739095/Streamfolder/Forked_ALNS_CMAB_MIP/ALNS/data/gen-ip002.mps.gz"
    # Create MIP instance
    run_banditalns(instance_path)
