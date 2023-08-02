import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from dlomix.constants import ALPHABET_UNMOD
from prosit_t.layers import (
    RegressorV2,
    Encoder,
    DecoderMeta,
    MetaEmbeddingSimple,
)

MAX_SEQUENCE_LENGTH = 30


class PrositTransformerWithMetaContextIntensityPredictor(tf.keras.Model):
    def __init__(
        self,
        embedding_output_dim=16,
        intermediate_dim=512,
        transformer_num_heads=4,
        seq_length=30,
        len_fion=6,
        vocab_dict=ALPHABET_UNMOD,
        dropout_rate=0.2,
        latent_dropout_rate=0.1,
        num_encoders=5,
        **kwargs
    ):
        super(PrositTransformerWithMetaContextIntensityPredictor, self).__init__()
        self.embeddings_count = len(vocab_dict) + 2
        self.max_ion = seq_length - 1

        self.string_lookup = preprocessing.StringLookup(
            vocabulary=list(vocab_dict.keys())
        )
        self.emb_meta = MetaEmbeddingSimple(d_model=embedding_output_dim)

        self.enc = Encoder(
            num_layers=num_encoders,
            d_model=embedding_output_dim,
            num_heads=transformer_num_heads,
            vocab_size=MAX_SEQUENCE_LENGTH,
            dff=256,
            dropout_rate=dropout_rate,
        )
        self.dec = DecoderMeta(
            num_layers=num_encoders,
            d_model=embedding_output_dim,
            num_heads=transformer_num_heads,
            dff=256,
            dropout_rate=dropout_rate,
        )
        self.flatten = tf.keras.layers.Flatten()
        self.regressor = RegressorV2(174)

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

        context = [precursor_charge_in, collision_energy_in]
        context = self.emb_meta(context)
        x = self.string_lookup(peptides_in)
        x = self.enc(x)
        x = self.dec(x=x, context=context)
        x = self.flatten(x)
        x = self.regressor(x)
        return x
