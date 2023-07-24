import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from dlomix.constants import ALPHABET_UNMOD
from prosit_t.layers import (
    RegressorV2,
    TransformerEncoder,
    PositionalEmbedding,
    DecoderMeta,
    MetaEmbeddingSimple,
)

MAX_SEQUENCE_LENGTH = 30


class PrositMetaContextIntensityPredictor(tf.keras.Model):
    def __init__(
        self,
        embedding_output_dim=16,
        seq_length=30,
        len_fion=6,
        vocab_dict=ALPHABET_UNMOD,
        dropout_rate=0.2,
        regressor_layer_size=512,
        num_heads=8,
        ff_dim=32,
        transformer_dropout=0.1,
        num_transformers=2,
        **kwargs
    ):
        super(PrositMetaContextIntensityPredictor, self).__init__()
        self.embeddings_count = len(vocab_dict) + 2
        self.max_ion = seq_length - 1

        self.string_lookup = preprocessing.StringLookup(
            vocabulary=list(vocab_dict.keys())
        )
        self.pos_embedding = PositionalEmbedding(
            self.embeddings_count, embedding_output_dim
        )
        self.emb_meta = MetaEmbeddingSimple(d_model=embedding_output_dim)

        self.transformer_encoder = TransformerEncoder(
            embedding_output_dim,
            num_heads,
            ff_dim,
            rate=transformer_dropout,
            num_transformers=num_transformers,
        )

        self.dec = DecoderMeta(
            num_layers=num_transformers,
            d_model=embedding_output_dim,
            num_heads=num_heads,
            dff=ff_dim,
            dropout_rate=transformer_dropout,
        )
        self.flatten = tf.keras.layers.Flatten()
        self.regressor_td = RegressorV2(len_fion * self.max_ion)

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
        x = self.pos_embedding(x)
        x = self.transformer_encoder(x)
        x = self.dec(x=x, context=context)
        x = self.flatten(x)
        x = self.regressor_td(x)
        return x
