"""Recurrent-based model classes wrapper.
"""

from nalp.models.generators import (GRUGenerator, LSTMGenerator, RMCGenerator,
                                    RNNGenerator)


class RecurrentModel:
    """A RecurrentModel class help users in selecting distinct recurrent architectures from the command line.

    """

    def __init__(self, obj):
        """Initialization method.

        Args:
            obj (Generator): A Generator-child instance.

        """

        # Creates a property to hold the class itself
        self.obj = obj


# Defines a model dictionary constant with the possible values
RECURRENT_MODEL = dict(
    gru=RecurrentModel(GRUGenerator),
    lstm=RecurrentModel(LSTMGenerator),
    rmc=RecurrentModel(RMCGenerator),
    rnn=RecurrentModel(RNNGenerator)
)


def get_recurrent_model(name):
    """Gets a model by its identifier.

    Args:
        name (str): RecurrentModel's identifier.

    Returns:
        An instance of the RecurrentModel class.

    """

    # Tries to invoke the method
    try:
        # Returns the corresponding object
        return RECURRENT_MODEL[name]

    # If object is not found
    except:
        # Raises a RuntimeError
        raise RuntimeError(
            f'Recurrent-based model {name} has not been specified yet.')
