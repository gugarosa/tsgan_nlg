"""Wrapper over Opytimizer pipeline.
"""

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace, TreeSpace


def start_opt(opt_name, opt, target, n_agents, n_variables, n_iterations, lb, ub, hyperparams):
    """Abstracts all Opytimizer's mechanisms into a single method.

    Args:
        opt_name (str): Name of optimizer.
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
    if opt_name == 'gp':
        # Creates a TreeSpace
        space = TreeSpace(n_agents, n_variables, lb, ub, 5, 2, 5, ['SUM', 'SUB', 'MUL', 'DIV'])

    # If optimization algorithm is something else
    else:
        # Creates the SearchSpace
        space = SearchSpace(n_agents, n_variables, lb, ub)

    # Creates the optimizer and function
    optimizer = opt(hyperparams)
    function = Function(target)

    # Bundles every piece into Opytimizer class
    task = Opytimizer(space, optimizer, function)

    # Initializes the task
    task.start(n_iterations)

    return task.history
