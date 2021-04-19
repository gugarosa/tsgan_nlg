"""Heuristic algorithms wrapper.
"""

from opytimizer.optimizers.evolutionary import gp
from opytimizer.optimizers.swarm import ba, cs, fa, pso


class Heuristic:
    """An Heuristic class help users in selecting distinct optimization heuristics from the command line.

    """

    def __init__(self, obj, hyperparams):
        """Initialization method.

        Args:
            obj (Optimizer): An Optimizer-child instance.
            hyperparams (dict): Meta-heuristic hyperparams.

        """

        # Creates a property to hold the class itself
        self.obj = obj

        # Creates a property to hold the hyperparams
        self.hyperparams = hyperparams


# Defines an heuristic dictionary constant with the possible values
HEURISTIC = dict(
    ba=Heuristic(ba.BA, dict(f_min=0, f_max=2, A=0.5, r=0.5)),
    cs=Heuristic(cs.CS, dict(alpha=0.3, beta=1.5, p=0.2)),
    fa=Heuristic(fa.FA, dict(alpha=0.5, beta=0.2, gamma=1.0)),
    gp=Heuristic(gp.GP, dict(p_reproduction=0.25, p_mutation=0.1,
                             p_crossover=0.2, prunning_ratio=0.0)),
    pso=Heuristic(pso.PSO, dict(w=0.7, c1=1.7, c2=1.7))
)


def get_heuristic(name):
    """Gets an heuristic by its identifier.

    Args:
        name (str): Heuristic's identifier.

    Returns:
        An instance of the Heuristic class.

    """

    # Tries to invoke the method
    try:
        # Returns the corresponding object
        return HEURISTIC[name]

    # If object is not found
    except Exception:
        # Raises a RuntimeError
        print(f'Heuristic {name} has not been specified yet.')
        raise
