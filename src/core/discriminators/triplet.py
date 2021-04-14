"""Triplet Loss-based Discriminator.
"""

import nalp.utils.logging as l
from dualing.models import TripletSiamese
from dualing.models.base import LSTM

logger = l.get_logger(__name__)


class TripletDiscriminator(TripletSiamese):
    """A TripletDiscriminator class is the one in charge of a
    Triplet-based discriminative implementation.

    References:
        Not published yet.

    """

    def __init__(self, vocab_size=1, embedding_size=32, hidden_size=64,
                 loss='hard', margin=0.5, soft=False, distance_metric='L2'):
        """Initialization method.

        Args:
            vocab_size (int): Vocabulary size.
            embedding_size (int): Embedding layer units.
            hidden_size (int): Hidden layer units.
            loss (str): Whether network should use hard or semi-hard negative mining.
            margin (float): Radius around the embedding space.
            soft (bool): Whether network should use soft margin or not.
            distance_metric (str): Distance metric.

        """

        logger.info('Overriding class: Discriminator -> TripletDiscriminator.')

        # Defines the base architecture
        base = LSTM(vocab_size=vocab_size, embedding_size=embedding_size, hidden_size=hidden_size)

        # Overrides its parent class with any custom arguments if needed
        super(TripletDiscriminator, self).__init__(base=base, loss=loss, margin=margin,
                                                   soft=soft, distance_metric=distance_metric,
                                                   name='D_Triplet')

        logger.info('Class overrided.')
