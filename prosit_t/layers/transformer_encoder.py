import tensorflow as tf
import keras_nlp


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        intermediate_dim=512,
        transformer_num_heads=4,
        mh_num_heads=4,
        key_dim=4,
        dropout_rate=0.2,
        layer_norm_epsilon=1e-5,
        regressor_layer_size=512,
        num_encoders=1,
    ):
        super(TransformerEncoder, self).__init__()
        self.num_encoders = num_encoders
        self.transformer_encoders = [
            keras_nlp.layers.TransformerEncoder(
                intermediate_dim=intermediate_dim,
                num_heads=transformer_num_heads,
                dropout=dropout_rate,
                layer_norm_epsilon=layer_norm_epsilon,
            )
            for _ in range(num_encoders)
        ]

    def call(self, x, **kwargs):
        for i in range(self.num_encoders):
            x = self.transformer_encoders[i](x)

        return x
