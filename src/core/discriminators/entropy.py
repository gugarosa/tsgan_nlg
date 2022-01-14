"""Cross-Entropy Loss-based Discriminator.
"""

import nalp.utils.logging as l
from dualing.models import CrossEntropySiamese
from dualing.models.base import RNN, LSTM, GRU

logger = l.get_logger(__name__)


class EntropyDiscriminator(CrossEntropySiamese):
    """An EntropyDiscriminator class is the one in charge of a
    Cross-Entropy-based discriminative implementation.

    References:
        Not published yet.

    """

    def __init__(self, vocab_size=1, embedding_size=32, hidden_size=64, distance_metric='concat'):
        """Initialization method.

        Args:
            vocab_size (int): Vocabulary size.
            embedding_size (int): Embedding layer units.
            hidden_size (int): Hidden layer units.
            distance_metric (str): Distance metric.

        """

        logger.info('Overriding class: Discriminator -> EntropyDiscriminator.')

        # Defines the base architecture
        base = LSTM(vocab_size=vocab_size, embedding_size=embedding_size, hidden_size=hidden_size)

        # Overrides its parent class with any custom arguments if needed
        super(EntropyDiscriminator, self).__init__(base=base, distance_metric=distance_metric,
                                                   name='D_Entropy')

        logger.info('Class overrided.')
