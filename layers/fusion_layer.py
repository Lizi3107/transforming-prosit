import tensorflow as tf

class FusionLayer(tf.keras.layer):
    def __init__(
        self,
        max_ion=29,
    ):
        super(FusionLayer, self).__init__()
        self.multiply = tf.keras.layers.Multiply(name="add_meta")
        self.repeat = tf.keras.layers.RepeatVector(self.max_ion, name="repeat")

    def call(self, inputs, **kwargs):
        x = self.multiply(inputs)
        x = self.repeat(x)

        return x
