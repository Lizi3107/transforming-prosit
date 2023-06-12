import tensorflow as tf


class RegressorV2(tf.keras.layers.Layer):
    def __init__(
        self,
        output_shape=174,
    ):
        super(RegressorV2, self).__init__()
        self.dense = tf.keras.layers.Dense(output_shape, name="dense")
        self.relu = tf.keras.layers.LeakyReLU(name="activation")
        self.flatten = tf.keras.layers.Flatten(name="out")

    def call(self, inputs, **kwargs):
        x = self.dense(inputs)
        x = self.relu(x)
        x = self.flatten(x)

        return x
