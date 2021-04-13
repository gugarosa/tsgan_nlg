"""Triplet Loss-based Discriminator.
"""

import nalp.utils.logging as l
from dualing.models import TripletSiamese
from dualing.models.base import CNN, MLP

logger = l.get_logger(__name__)


class TripletDiscriminator(TripletSiamese):
    """A TripletDiscriminator class is the one in charge of a
    Triplet-based discriminative implementation.

    References:
        Not published yet.

    """

    def __init__(self, base=None, loss='hard', margin=0.5, soft=False, distance_metric='L2'):
        """Initialization method.

        Args:
            base (Base): Twin architecture.
            loss (str): Whether network should use hard or semi-hard negative mining.
            margin (float): Radius around the embedding space.
            soft (bool): Whether network should use soft margin or not.
            distance_metric (str): Distance metric.

        """

        logger.info('Overriding class: Discriminator -> TripletDiscriminator.')

        # base = CNN(n_blocks=3, init_kernel=5, n_output=128)
        base = MLP(n_hidden=(512, 256, 128))

        # Overrides its parent class with any custom arguments if needed
        super(TripletDiscriminator, self).__init__(base=base, loss=loss, margin=margin,
                                                   soft=soft, distance_metric=distance_metric,
                                                   name='D_Triplet')

        
        logger.info('Class overrided.')
