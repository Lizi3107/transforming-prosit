import tensorflow as tf
import keras_nlp


class SequenceEncoderNoGRU(tf.keras.layers.Layer):
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
        super(SequenceEncoderNoGRU, self).__init__()

        self.transformer_encoder = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=intermediate_dim,
            num_heads=transformer_num_heads,
            dropout=dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
        )

    def call(self, inputs, **kwargs):
        x = self.transformer_encoder(inputs)

        return x
