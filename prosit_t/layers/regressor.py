import tensorflow as tf


class Regressor(tf.keras.layers.Layer):
    def __init__(
        self,
        len_fion=6,
    ):
        super(Regressor, self).__init__()
        self.dense = tf.keras.layers.Dense(len_fion, name="dense")
        self.relu = tf.keras.layers.LeakyReLU(name="activation")
        self.flatten = tf.keras.layers.Flatten(name="out")

    def call(self, inputs, **kwargs):
        x = self.dense(inputs)
        x = self.relu(x)
        x = self.flatten(x)

        return x
