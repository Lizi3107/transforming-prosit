import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from dlomix.constants import ALPHABET_UNMOD
from dlomix.layers.attention import AttentionLayer
from prosit_t.layers import (
    MetaEncoder,
    FusionLayer,
    TransformerDecoder,
    RegressorV2,
    TransformerEncoder,
    PositionalEmbedding,
)


class PrositTransformerIntensityPredictor(tf.keras.Model):
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
        layer_norm_epsilon=1e-5,
        num_encoders=5,
        **kwargs
    ):
        super(PrositTransformerIntensityPredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of aa)
        self.embeddings_count = len(vocab_dict) + 2

        # maximum number of fragment ions
        self.max_ion = seq_length - 1

        self.meta_encoder = MetaEncoder(embedding_output_dim, dropout_rate)

        self.string_lookup = preprocessing.StringLookup(
            vocabulary=list(vocab_dict.keys())
        )

        self.pos_embedding = PositionalEmbedding(
            self.embeddings_count, embedding_output_dim
        )

        self.attention = AttentionLayer(name="encoder_att")
        self.fusion_layer = FusionLayer(self.max_ion)
        self.sequence_encoder = TransformerEncoder(
            intermediate_dim,
            transformer_num_heads,
            mh_num_heads,
            key_dim,
            dropout_rate,
            layer_norm_epsilon,
            num_encoders,
            normalize_first=True,
        )
        self.decoder = TransformerDecoder(
            intermediate_dim,
            transformer_num_heads,
            mh_num_heads,
            key_dim,
            dropout_rate,
            layer_norm_epsilon,
            self.max_ion,
            num_encoders,
            normalize_first=True,
        )

        self.flatten_2 = tf.keras.layers.Flatten()
        # self.extract_last_token = tf.keras.layers.Lambda(lambda x: x[:, -1, :])

        self.regressor = RegressorV2(len_fion * self.max_ion)
        self.relu = tf.keras.layers.ReLU()
        self.flatten_3 = tf.keras.layers.Flatten(name="out")

        self.before_dense = tf.Variable(
            initial_value=tf.zeros((64, 1856)), trainable=False
        )
        self.after_dense = tf.Variable(
            initial_value=tf.zeros((64, 174)), trainable=False
        )
        self.after_relu = tf.Variable(
            initial_value=tf.zeros((64, 174)), trainable=False
        )

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

    def call(self, inputs, training=None, **kwargs):
        peptides_in = inputs["sequence"]
        collision_energy_in = inputs["collision_energy"]
        precursor_charge_in = inputs["precursor_charge"]
        encoded_meta = self.meta_encoder([collision_energy_in, precursor_charge_in])
        x = self.string_lookup(peptides_in)
        x = self.pos_embedding(x)
        x = self.attention(x)
        x = self.fusion_layer([x, encoded_meta])
        x = self.sequence_encoder(x)
        x = self.decoder(x)
        x = self.flatten_2(x)
        if training is False:
            self.before_dense.assign(x)
        x = self.regressor(x)
        if training is False:
            self.after_dense.assign(x)
        x = self.relu(x)
        if training is False:
            self.after_relu.assign(x)
        x = self.flatten_3(x)
        return x
