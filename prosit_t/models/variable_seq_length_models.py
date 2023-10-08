import tensorflow as tf
from dlomix.constants import ALPHABET_UNMOD
from tensorflow.keras.layers.experimental import preprocessing
from prosit_t.layers import (
    MetaEncoder,
    PositionalEmbedding,
    TransformerEncoder,
    CrossAttention,
)


class ProstTransformerDynamicLen(tf.keras.Model):
    def __init__(
        self,
        embedding_output_dim=16,
        vocab_dict=ALPHABET_UNMOD,
        len_fion=6,
        dropout_rate=0.2,
        num_heads=8,
        ff_dim=32,
        transformer_dropout=0.1,
        num_transformers=2,
        dense_dim_factor=4,
        **kwargs,
    ):
        super(ProstTransformerDynamicLen, self).__init__()

        self.embeddings_count = len(vocab_dict) + 2
        self.embed = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_output_dim,
            mask_zero=False,
        )
        self.flatten = tf.keras.layers.Flatten()
        self.tdense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len_fion))

        self.string_lookup = preprocessing.StringLookup(
            vocabulary=list(vocab_dict.keys())
        )
        self.pos_embedding = PositionalEmbedding(
            self.embeddings_count, embedding_output_dim
        )
        self.meta_encoder = MetaEncoder(
            embedding_output_dim * dense_dim_factor, dropout_rate
        )
        self.transformer_encoder = TransformerEncoder(
            embedding_output_dim,
            num_heads,
            ff_dim,
            rate=transformer_dropout,
            num_transformers=num_transformers,
        )
        self.reshape = tf.keras.layers.Reshape(
            [1, embedding_output_dim * dense_dim_factor]
        )
        self.cross_att = CrossAttention(
            num_heads=num_heads,
            key_dim=16,
            dropout=0,
        )
        self.drop = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])

    def call(self, inputs, **kwargs):
        peptides_in = inputs["sequence"]
        collision_energy_in = inputs["collision_energy"]
        precursor_charge_in = inputs["precursor_charge"]
        encoded_meta = self.meta_encoder([collision_energy_in, precursor_charge_in])
        encoded_meta = self.reshape(encoded_meta)
        x = self.string_lookup(peptides_in)
        x = self.pos_embedding(x)
        x = self.transformer_encoder(x)  # batch, 3, 64
        x = self.cross_att(x=x, context=encoded_meta)  # batch, 3, 64
        x = self.drop(x)  # batch, 2, 64
        x = self.tdense(x)  # batch, 2, 6
        x = self.flatten(x)
        return x


class ProstTransformerDynamicLenPooling(tf.keras.Model):
    def __init__(
        self,
        embedding_output_dim=16,
        vocab_dict=ALPHABET_UNMOD,
        len_fion=6,
        dropout_rate=0.2,
        num_heads=8,
        ff_dim=32,
        transformer_dropout=0.1,
        num_transformers=2,
        dense_dim_factor=4,
        **kwargs,
    ):
        super(ProstTransformerDynamicLenPooling, self).__init__()

        self.embeddings_count = len(vocab_dict) + 2
        self.embed = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_output_dim,
            mask_zero=False,
        )
        self.flatten = tf.keras.layers.Flatten()
        self.tdense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len_fion))

        self.string_lookup = preprocessing.StringLookup(
            vocabulary=list(vocab_dict.keys())
        )
        self.pos_embedding = PositionalEmbedding(
            self.embeddings_count, embedding_output_dim
        )
        self.meta_encoder = MetaEncoder(
            embedding_output_dim * dense_dim_factor, dropout_rate
        )
        self.transformer_encoder = TransformerEncoder(
            embedding_output_dim,
            num_heads,
            ff_dim,
            rate=transformer_dropout,
            num_transformers=num_transformers,
        )
        self.reshape = tf.keras.layers.Reshape(
            [1, embedding_output_dim * dense_dim_factor]
        )
        self.cross_att = CrossAttention(
            num_heads=num_heads,
            key_dim=16,
            dropout=0,
        )
        self.pool = tf.keras.layers.AveragePooling1D(
            pool_size=2, strides=1, padding="valid"
        )

    def call(self, inputs, **kwargs):
        peptides_in = inputs["sequence"]
        collision_energy_in = inputs["collision_energy"]
        precursor_charge_in = inputs["precursor_charge"]

        encoded_meta = self.meta_encoder([collision_energy_in, precursor_charge_in])
        encoded_meta = self.reshape(encoded_meta)
        x = self.string_lookup(peptides_in)
        x = self.pos_embedding(x)
        x = self.transformer_encoder(x)  # batch, 3, 64
        x = self.cross_att(x=x, context=encoded_meta)  # batch, 3, 64
        x = self.pool(x)  # batch, 2, 64
        x = self.tdense(x)  # batch, 2, 6
        x = self.flatten(x)
        return x
