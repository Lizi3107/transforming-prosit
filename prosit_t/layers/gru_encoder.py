import tensorflow as tf


class GRUEncoder(tf.keras.layers.Layer):
    def __init__(self, recurrent_layers_sizes=512, dropout_rate=0.2, max_ion=29):
        super(GRUEncoder, self).__init__()
        self.bi_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=recurrent_layers_sizes[0], return_sequences=True)
        )
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.gru = tf.keras.layers.GRU(
            units=recurrent_layers_sizes[1], return_sequences=True
        )
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, **kwargs):
        x = self.bi_gru(inputs)
        x = self.dropout_1(x)
        x = self.gru(x)
        x = self.dropout_2(x)

        return x
