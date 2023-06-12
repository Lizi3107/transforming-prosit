import tensorflow as tf
from prosit_t.layers import PositionalEmbedding, GlobalSelfAttention, CrossAttention, CausalSelfAttention, FeedForward

class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)
        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

class MetaEmbeddingSimple(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.projection_head = tf.keras.layers.Dense(d_model, activation='relu')
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.concat = tf.keras.layers.Concatenate()
        self.reshape = tf.keras.layers.Reshape([1, d_model])

    def call(self, x):
        charge, collision_energy = x
        x = self.concat([charge, collision_energy])
        x = self.projection_head(x)
        x = self.reshape(x)
        x = self.layernorm(x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x
    
class DecoderMeta(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
               dropout_rate=0.1):
        super(DecoderMeta, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers


        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, context)
        return x