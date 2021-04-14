"""Contrastive Loss-based Discriminator.
"""

import nalp.utils.logging as l
from dualing.models import ContrastiveSiamese
from dualing.models.base import LSTM

logger = l.get_logger(__name__)


class ContrastiveDiscriminator(ContrastiveSiamese):
    """A ContrastiveDiscriminator class is the one in charge of a
    Contrastive-based discriminative implementation.

    References:
        Not published yet.

    """

    def __init__(self, vocab_size=1, embedding_size=32, hidden_size=64, margin=1.0, distance_metric='L2'):
        """Initialization method.

        Args:
            vocab_size (int): Vocabulary size.
            embedding_size (int): Embedding layer units.
            hidden_size (int): Hidden layer units.
            margin (float): Radius around the embedding space.
            distance_metric (str): Distance metric.

        """

        logger.info('Overriding class: Discriminator -> ContrastiveDiscriminator.')

        # Defines the base architecture
        base = LSTM(vocab_size=vocab_size, embedding_size=embedding_size, hidden_size=hidden_size)

        # Overrides its parent class with any custom arguments if needed
        super(ContrastiveDiscriminator, self).__init__(base=base, margin=margin,
                                                       distance_metric=distance_metric,
                                                       name='D_contrastive')

        logger.info('Class overrided.')
