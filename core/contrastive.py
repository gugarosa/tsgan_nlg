"""Contrastive-based discriminator.
"""

import nalp.utils.logging as l

from dualing.models.base import CNN, MLP
from dualing.models import ContrastiveSiamese

logger = l.get_logger(__name__)


class ContrastiveDiscriminator(ContrastiveSiamese):
    """A ContrastiveDiscriminator class is the one in charge of a
    Contrastive-based discriminative implementation.

    References:
        Not published yet.

    """

    def __init__(self, base=None, margin=1.0, distance_metric='L2'):
        """Initialization method.

        Args:
            base (Base): Twin architecture.
            margin (float): Radius around the embedding space.
            distance_metric (str): Distance metric.

        """

        logger.info('Overriding class: Discriminator -> ContrastiveDiscriminator.')

        #
        # base = CNN(n_blocks=3, init_kernel=5, n_output=128)
        base = MLP(n_hidden=(512, 256, 128))

        # Overrides its parent class with any custom arguments if needed
        super(ContrastiveDiscriminator, self).__init__(base=base, margin=margin,
                                                       distance_metric=distance_metric,
                                                       name='D_contrastive')

        
        logger.info('Class overrided.')
