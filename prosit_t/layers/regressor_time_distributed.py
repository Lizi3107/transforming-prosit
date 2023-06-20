import tensorflow as tf


class RegressorTimeDistributed(tf.keras.layers.Layer):
    def __init__(
        self,
        len_fion=6,
    ):
        super(RegressorTimeDistributed, self).__init__()
        self.td_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(len_fion), name="time_dense"
        )
        self.relu = tf.keras.layers.LeakyReLU(name="activation")
        self.flatten = tf.keras.layers.Flatten(name="out")

    def call(self, inputs, **kwargs):
        x = self.td_dense(inputs)
        x = self.relu(x)
        x = self.flatten(x)

        return x
