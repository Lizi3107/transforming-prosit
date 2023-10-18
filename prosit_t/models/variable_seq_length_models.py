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
        meta_embedding_dim=64,
        vocab_dict=ALPHABET_UNMOD,
        len_fion=6,
        dropout_rate=0.2,
        num_heads=8,
        ff_dim=32,
        transformer_dropout=0.1,
        num_transformers=2,
        **kwargs,
    ):
        super(ProstTransformerDynamicLen, self).__init__()

        self.embeddings_count = len(vocab_dict) + 2
        self.string_lookup = preprocessing.StringLookup(
            vocabulary=list(vocab_dict.keys())
        )
        self.pos_embedding = PositionalEmbedding(
            self.embeddings_count, embedding_output_dim
        )
        self.meta_encoder = MetaEncoder(meta_embedding_dim, dropout_rate)
        self.reshape = tf.keras.layers.Reshape([1, meta_embedding_dim])
        self.transformer_encoder = TransformerEncoder(
            embedding_output_dim,
            num_heads,
            ff_dim,
            rate=transformer_dropout,
            num_transformers=num_transformers,
        )
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.cross_att = CrossAttention(
            num_heads=num_heads,
            key_dim=16,
            dropout=0,
        )
        self.tdense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len_fion))
        self.flatten = tf.keras.layers.Flatten()

    @classmethod
    def call(self, inputs, **kwargs):
        raise NotImplementedError


class ProstTransformerDynamicLenDropLast(ProstTransformerDynamicLen):
    def __init__(
        self,
        embedding_output_dim=16,
        meta_embedding_dim=64,
        vocab_dict=ALPHABET_UNMOD,
        len_fion=6,
        dropout_rate=0.2,
        num_heads=8,
        ff_dim=32,
        transformer_dropout=0.1,
        num_transformers=2,
        **kwargs,
    ):
        super(ProstTransformerDynamicLenDropLast, self).__init__(
            embedding_output_dim,
            meta_embedding_dim,
            vocab_dict,
            len_fion,
            dropout_rate,
            num_heads,
            ff_dim,
            transformer_dropout,
            num_transformers,
            **kwargs,
        )

        self.drop_last = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])

    def call(self, inputs, **kwargs):
        peptides_in = inputs["sequence"]
        collision_energy_in = inputs["collision_energy"]
        precursor_charge_in = inputs["precursor_charge"]
        encoded_meta = self.meta_encoder([collision_energy_in, precursor_charge_in])
        encoded_meta = self.reshape(encoded_meta)
        x = self.string_lookup(peptides_in)
        x = self.pos_embedding(x)
        x = self.transformer_encoder(x)
        x = self.leaky_relu(x)
        x = self.cross_att(x=x, context=encoded_meta)
        x = self.leaky_relu(x)
        x = self.drop_last(x)
        x = self.tdense(x)
        x = self.flatten(x)
        return x


class ProstTransformerDynamicLenDropFirst(ProstTransformerDynamicLen):
    def __init__(
        self,
        embedding_output_dim=16,
        meta_embedding_dim=64,
        vocab_dict=ALPHABET_UNMOD,
        len_fion=6,
        dropout_rate=0.2,
        num_heads=8,
        ff_dim=32,
        transformer_dropout=0.1,
        num_transformers=2,
        **kwargs,
    ):
        super(ProstTransformerDynamicLenDropFirst, self).__init__(
            embedding_output_dim,
            meta_embedding_dim,
            vocab_dict,
            len_fion,
            dropout_rate,
            num_heads,
            ff_dim,
            transformer_dropout,
            num_transformers,
            **kwargs,
        )

        self.drop_first = tf.keras.layers.Lambda(lambda x: x[:, 1:, :])

    def call(self, inputs, **kwargs):
        peptides_in = inputs["sequence"]
        collision_energy_in = inputs["collision_energy"]
        precursor_charge_in = inputs["precursor_charge"]
        encoded_meta = self.meta_encoder([collision_energy_in, precursor_charge_in])
        encoded_meta = self.reshape(encoded_meta)
        x = self.string_lookup(peptides_in)
        x = self.pos_embedding(x)
        x = self.transformer_encoder(x)
        x = self.leaky_relu(x)
        x = self.cross_att(x=x, context=encoded_meta)
        x = self.leaky_relu(x)
        x = self.drop_first(x)
        x = self.tdense(x)
        x = self.flatten(x)
        return x


class ProstTransformerDynamicLenPooling(ProstTransformerDynamicLen):
    def __init__(
        self,
        embedding_output_dim=16,
        meta_embedding_dim=64,
        vocab_dict=ALPHABET_UNMOD,
        len_fion=6,
        dropout_rate=0.2,
        num_heads=8,
        ff_dim=32,
        transformer_dropout=0.1,
        num_transformers=2,
        **kwargs,
    ):
        super(ProstTransformerDynamicLenPooling, self).__init__(
            embedding_output_dim,
            meta_embedding_dim,
            vocab_dict,
            len_fion,
            dropout_rate,
            num_heads,
            ff_dim,
            transformer_dropout,
            num_transformers,
            **kwargs,
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
        x = self.transformer_encoder(x)
        x = self.leaky_relu(x)
        x = self.cross_att(x=x, context=encoded_meta)
        x = self.leaky_relu(x)
        x = self.pool(x)
        x = self.tdense(x)
        x = self.flatten(x)
        return x
