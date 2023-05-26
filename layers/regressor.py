import tensorflow as tf

class Regressor(tf.keras.layer):
    def __init__(
        self,
        regressor_layer_size=512,
        latent_dropout_rate=0.2,
    ):
        super(Regressor, self).__init__()
        self.dense = tf.keras.layers.Dense(regressor_layer_size, activation="relu")
        self.dropout = tf.keras.layers.Dropout(rate=latent_dropout_rate)

    def call(self, inputs, **kwargs):
        x = self.dense(inputs)
        x = self.dropout(x)

        return x
