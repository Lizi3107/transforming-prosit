import tensorflow as tf
from dlomix.constants import ALPHABET_UNMOD


class TestModel(tf.keras.Model):
    def __init__(
        self,
        embedding_output_dim=16,
        vocab_dict=ALPHABET_UNMOD,
        len_fion=6,
        **kwargs,
    ):
        super(TestModel, self).__init__()

        self.embeddings_count = len(vocab_dict) + 2
        self.embed = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_output_dim,
            mask_zero=False,
        )
        self.flatten = tf.keras.layers.Flatten()
        self.tdense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len_fion))

    def call(self, inputs, **kwargs):
        peptides_in = inputs["sequence"]
        collision_energy_in = inputs["collision_energy"]
        precursor_charge_in = inputs["precursor_charge"]

        x = self.embed(peptides_in)
        x = self.tdense(x)
        x = self.flatten(x)
        return x
