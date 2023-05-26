import tensorflow as tf
from dlomix.layers.attention import DecoderAttentionLayer

class GRUDecoder(tf.keras.layer):
    def __init__(self, regressor_layer_size=512, dropout_rate=0.2, max_ion=29):
        super(GRUDecoder, self).__init__()
        self.max_ion
        self.gru = tf.keras.layers.GRU(
            units=regressor_layer_size,
            return_sequences=True,
            name="decoder",
        )
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, **kwargs):
        x = self.gru(inputs)
        x = self.dropout(x)
        x = DecoderAttentionLayer(self.max_ion)(x)

        return x
