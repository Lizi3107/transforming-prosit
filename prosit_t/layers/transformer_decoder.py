import tensorflow as tf
from dlomix.layers.attention import DecoderAttentionLayer
import keras_nlp


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        intermediate_dim=512,
        transformer_num_heads=4,
        mh_num_heads=4,
        key_dim=4,
        dropout_rate=0.2,
        layer_norm_epsilon=1e-5,
        max_ion=29,
        num_encoders=1,
        normalize_first=True,
    ):
        super(TransformerDecoder, self).__init__()
        self.num_encoders = num_encoders
        self.transformer_encoders = [
            keras_nlp.layers.TransformerEncoder(
                intermediate_dim=intermediate_dim,
                num_heads=transformer_num_heads,
                dropout=dropout_rate,
                layer_norm_epsilon=layer_norm_epsilon,
                normalize_first=normalize_first,
            )
            for _ in range(num_encoders)
        ]
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.decoder_att = DecoderAttentionLayer(max_ion)

    def call(self, x, **kwargs):
        for i in range(self.num_encoders):
            x = self.transformer_encoders[i](x)
            x = self.dropout(x)
            x = self.decoder_att(x)

        return x
