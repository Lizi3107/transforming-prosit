import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from dlomix.constants import ALPHABET_UNMOD
from dlomix.layers.attention import AttentionLayer
from prosit_t.layers import (
    MetaEncoder,
    FusionLayer,
    RegressorTimeDistributed,
    SequenceEncoderTransformerGRU,
    GRUDecoder,
)


class PrositIntensityPredictor(tf.keras.Model):
    def __init__(
        self,
        embedding_output_dim=16,
        intermediate_dim=512,
        transformer_num_heads=4,
        mh_num_heads=4,
        key_dim=4,
        seq_length=30,
        len_fion=6,
        vocab_dict=ALPHABET_UNMOD,
        dropout_rate=0.2,
        latent_dropout_rate=0.1,
        regressor_layer_size=512,
        layer_norm_epsilon=1e-5,
    ):
        super(PrositIntensityPredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of aa)
        self.embeddings_count = len(vocab_dict) + 2

        # maximum number of fragment ions
        self.max_ion = seq_length - 1

        self.string_lookup = preprocessing.StringLookup(
            vocabulary=list(vocab_dict.keys())
        )

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_output_dim,
            input_length=seq_length,
        )

        self.meta_encoder = MetaEncoder(regressor_layer_size, dropout_rate)

        self.sequence_encoder = SequenceEncoderTransformerGRU(
            intermediate_dim,
            transformer_num_heads,
            mh_num_heads,
            key_dim,
            dropout_rate,
            layer_norm_epsilon,
            regressor_layer_size,
        )
        self.attention = AttentionLayer(name="encoder_att")
        self.fusion_layer = FusionLayer(self.max_ion)
        self.decoder = GRUDecoder(regressor_layer_size, dropout_rate, self.max_ion)
        self.regressor_td = RegressorTimeDistributed(len_fion)

    def summary(self):
        in_sequence = tf.keras.layers.Input(shape=(30,))
        in_collision_energy = tf.keras.layers.Input(shape=(1,))
        in_precursor_charge = tf.keras.layers.Input(shape=(6,))
        outputs = self.call(
            {
                "sequence": in_sequence,
                "collision_energy": in_collision_energy,
                "precursor_charge": in_precursor_charge,
            }
        )
        return tf.keras.Model(
            inputs=[in_sequence, in_collision_energy, in_precursor_charge],
            outputs=outputs,
        ).summary()

    def call(self, inputs, **kwargs):
        peptides_in = inputs["sequence"]
        collision_energy_in = inputs["collision_energy"]
        precursor_charge_in = inputs["precursor_charge"]

        encoded_meta = self.meta_encoder([collision_energy_in, precursor_charge_in])
        x = self.string_lookup(peptides_in)
        x = self.embedding(x)
        x = self.sequence_encoder(x)
        x = self.attention(x)
        x = self.fusion_layer([x, encoded_meta])
        x = self.decoder(x)
        x = self.regressor_td(x)
        return x
