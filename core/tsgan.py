"""Text-Similarity Generative Adversarial Network.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

import nalp.utils.constants as c
import nalp.utils.logging as l
from nalp.core import Adversarial
from nalp.models.generators import LSTMGenerator

from core.contrastive import ContrastiveDiscriminator

logger = l.get_logger(__name__)


class TSGAN(Adversarial):
    """A TSGAN class is the one in charge of Text-Similarity Generative Adversarial Networks implementation.

    References:
        Not published yet.

    """

    def __init__(self, encoder=None, vocab_size=1, max_length=1, embedding_size=32, hidden_size=64,
                 temperature=1):
        """Initialization method.

        Args:
            encoder (IntegerEncoder): An index to vocabulary encoder for the generator.
            vocab_size (int): The size of the vocabulary for both discriminator and generator.
            max_length (int): Maximum length of the sequences for the discriminator.
            embedding_size (int): The size of the embedding layer for both discriminator and generator.
            hidden_size (int): The amount of hidden neurons for the generator.
            temperature (float): Temperature value to sample the token.

        """

        logger.info('Overriding class: Adversarial -> TSGAN.')

        # Creating the discriminator network
        D = ContrastiveDiscriminator()

        # Creating the generator network
        G = LSTMGenerator(encoder, vocab_size, embedding_size, hidden_size)

        # Overrides its parent class with any custom arguments if needed
        super(TSGAN, self).__init__(D, G, name='TSGAN')

        # Defining a property for holding the vocabulary size
        self.vocab_size = vocab_size

        # Defining a property for holding the temperature
        self.T = temperature

        logger.info('Class overrided.')

    @property
    def vocab_size(self):
        """int: The size of the vocabulary.

        """

        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, vocab_size):
        self._vocab_size = vocab_size

    @property
    def T(self):
        """float: Temperature value to sample the token.

        """

        return self._T

    @T.setter
    def T(self, T):
        self._T = T

    def compile(self, pre_d_optimizer, pre_g_optimizer, d_optimizer, g_optimizer):
        """Main building method.

        Args:
            pre_d_optimizer (tf.keras.optimizers): An optimizer instance for pre-training the discrminator.
            pre_g_optimizer (tf.keras.optimizers): An optimizer instance for pre-training the generator.
            d_optimizer (tf.keras.optimizers): An optimizer instance for the discriminator.
            g_optimizer (tf.keras.optimizers): An optimizer instance for the generator.

        """

        # Compiles the discriminator
        self.D.compile(optimizer=pre_d_optimizer)

        # Creates optimizers for pre-training, discriminator and generator
        self.P_optimizer = pre_g_optimizer
        self.D_optimizer = d_optimizer
        self.G_optimizer = g_optimizer

        # Defining the loss function
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits

        # Defining both loss metrics
        self.D_loss = tf.metrics.Mean(name='D_loss')
        self.G_loss = tf.metrics.Mean(name='G_loss')

        # Storing losses as history keys
        self.history['pre_D_loss'] = []
        self.history['pre_G_loss'] = []
        self.history['D_loss'] = []
        self.history['G_loss'] = []

    @tf.function
    def G_pre_step(self, x, y):
        """Performs a single batch optimization pre-fitting step over the generator.

        Args:
            x (tf.tensor): A tensor containing the inputs.
            y (tf.tensor): A tensor containing the inputs' labels.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            # Calculate the predictions based on inputs
            preds = self.G(x)

            # Calculate the loss
            loss = tf.reduce_mean(self.loss(y, preds))

        # Calculate the gradient based on loss for each training variable
        gradients = tape.gradient(loss, self.G.trainable_variables)

        # Apply gradients using an optimizer
        self.P_optimizer.apply_gradients(
            zip(gradients, self.G.trainable_variables))

        # Updates the generator's loss state
        self.G_loss.update_state(loss)

    def pre_fit(self, batches, g_epochs=50, d_epochs=10):
        """Pre-trains the model.

        Args:
            batches (Dataset): Pre-training batches containing samples.
            g_epochs (int): The maximum number of pre-training generator epochs.
            d_epochs (int): The maximum number of pre-training discriminator epochs.

        """

        logger.info('Pre-fitting generator ...')

        # Gathering the amount of batches
        n_batches = tf.data.experimental.cardinality(batches).numpy()

        # Iterate through all generator epochs
        for e in range(g_epochs):
            logger.info('Epoch %d/%d', e+1, g_epochs)

            # Resetting state to further append losses
            self.G_loss.reset_states()

            # Defining a customized progress bar
            b = Progbar(n_batches, stateful_metrics=['loss(G)'])

            # Iterate through all possible pre-training batches
            for x_batch, y_batch in batches:
                # Performs the optimization step over the generator
                self.G_pre_step(x_batch, y_batch)

                # Adding corresponding values to the progress bar
                b.add(1, values=[('loss(G)', self.G_loss.result())])
         
            # Dump loss to history
            self.history['pre_G_loss'].append(self.G_loss.result().numpy())

            logger.to_file('Loss(G): %s', self.G_loss.result().numpy())

        logger.info('Pre-fitting discriminator ...')

        # Iterate through all discriminator epochs
        for e in range(d_epochs):
            logger.info('Epoch %d/%d', e+1, d_epochs)

            # Resetting state to further append losses
            self.D.loss_metric.reset_states()

            # Defining a customized progress bar
            b = Progbar(n_batches, stateful_metrics=['loss(D)'])

            # Iterate through all possible pre-training batches
            for x_batch, _ in batches:
                #
                x1_batch = tf.random.normal((4, 128))
                x2_batch = tf.random.normal((4, 128))
                y_batch = [1, 1, 1, 1]

                #
                self.D.step(x1_batch, x2_batch, y_batch)

                # Adding corresponding values to the progress bar
                b.add(1, values=[('loss(D)', self.D.loss_metric.result())])

            # Dump loss to history
            self.history['pre_D_loss'].append(self.D.loss_metric.result().numpy())

            logger.to_file('Loss(D): %s', self.D.loss_metric.result().numpy())
