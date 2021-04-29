"""Adversarial-based models wrapper.
"""

from nalp.models import GSGAN, MaliGAN, RelGAN, SeqGAN

from core import TSGANContrastive, TSGANEntropy, TSGANTriplet


class AdversarialModel:
    """An AdversarialModel class help users in selecting distinct adversarial architectures from the command line.

    """

    def __init__(self, obj):
        """Initialization method.

        Args:
            obj (Adversarial): An Adversarial-child instance.

        """

        # Creates a property to hold the class itself
        self.obj = obj


# Defines a model dictionary constant with the possible values
ADVERSARIAL_MODEL = dict(
    gsgan=AdversarialModel(GSGAN),
    maligan=AdversarialModel(MaliGAN),
    relgan=AdversarialModel(RelGAN),
    seqgan=AdversarialModel(SeqGAN),
    tsgan_contrastive=AdversarialModel(TSGANContrastive),
    tsgan_entropy=AdversarialModel(TSGANEntropy),
    tsgan_triplet=AdversarialModel(TSGANTriplet)
)


def get_adversarial_model(name):
    """Gets a model by its identifier.

    Args:
        name (str): AdversarialModel's identifier.

    Returns:
        An instance of the AdversarialModel class.

    """

    # Tries to invoke the method
    try:
        # Returns the corresponding object
        return ADVERSARIAL_MODEL[name]

    # If object is not found
    except Exception:
        # Raises a RuntimeError
        print(f'Adversarial-based model {name} has not been specified yet.')
        raise
