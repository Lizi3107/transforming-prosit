import tensorflow as tf
import keras_nlp


class SequenceEncoderTransformerGRU(tf.keras.layers.Layer):
    def __init__(
        self,
        intermediate_dim=512,
        transformer_num_heads=4,
        mh_num_heads=4,
        key_dim=4,
        dropout_rate=0.2,
        layer_norm_epsilon=1e-5,
        regressor_layer_size=512,
    ):
        super(SequenceEncoderTransformerGRU, self).__init__()

        self.transformer_encoder = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=intermediate_dim,
            num_heads=transformer_num_heads,
            dropout=dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
        )
        self.mh_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=mh_num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
            output_shape=regressor_layer_size,
        )
        self.sequence_gru = tf.keras.Sequential(
            [
                tf.keras.layers.GRU(units=regressor_layer_size, return_sequences=True),
                tf.keras.layers.Dropout(rate=dropout_rate),
            ]
        )

    def call(self, inputs, **kwargs):
        x = self.transformer_encoder(inputs)
        x = self.mh_attention(x, x)
        x = self.sequence_gru(x)

        return x
