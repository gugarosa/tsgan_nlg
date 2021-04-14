"""Wrapper over Opytimizer pipeline.
"""

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.core.optimizer import Optimizer
from opytimizer.spaces.search import SearchSpace
from opytimizer.spaces.tree import TreeSpace


def start_opt(opt, target, n_agents, n_variables, n_iterations, lb, ub, hyperparams):
    """Abstracts all Opytimizer's mechanisms into a single method.

    Args:
        opt (Optimizer): An Optimizer-child class.
        target (callable): The method to be optimized.
        n_agents (int): Number of agents.
        n_variables (int): Number of variables.
        n_iterations (int): Number of iterations.
        lb (list): List of lower bounds.
        ub (list): List of upper bounds.
        hyperparams (dict): Dictionary of hyperparameters.

    Returns:
        A History object containing all optimization information.

    """

    # Checks if optimization algorithm is GP
    if opt.algorithm == 'gp':
        # Creates a TreeSpace
        space = TreeSpace(n_trees=n_agents, n_terminals=5, n_variables=n_variables,
                          n_iterations=n_iterations, min_depth=2, max_depth=5,
                          functions=['SUM', 'SUB', 'MUL', 'DIV'], lower_bound=lb, upper_bound=ub)
    
    # If optimization algorithm is something else
    else:
        # Creates the SearchSpace
        space = SearchSpace(n_agents=n_agents, n_variables=n_variables, n_iterations=n_iterations,
                            lower_bound=lb, upper_bound=ub)

    # Creates the Optimizer
    optimizer = opt(hyperparams=hyperparams)

    # Creates the Function
    function = Function(pointer=target)

    # Creates the optimization task
    task = Opytimizer(space=space, optimizer=optimizer, function=function)

    # Initializes task
    history = task.start(store_best_only=True)

    return history
