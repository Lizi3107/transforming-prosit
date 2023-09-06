import tensorflow as tf
from dlomix.constants import ALPHABET_UNMOD


class TestModel(tf.keras.Model):
    def __init__(
        self,
        embedding_output_dim=16,
        vocab_dict=ALPHABET_UNMOD,
        **kwargs,
    ):
        super(TestModel, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of aa)
        self.embeddings_count = len(vocab_dict) + 2
        self.embed = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_output_dim,
            mask_zero=False,
        )

    # def build(self, input_shape):
    #     super(TestModel, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, tf.RaggedTensor):
            inputs = tf.RaggedTensor.from_tensor(inputs)
        row_splits = inputs.row_splits
        sample_len = row_splits[1] - row_splits[0]
        tf.print("Row Splits:", row_splits)
        tf.print("sample_len:", sample_len)

        x = self.embed(inputs)
        # dense = tf.keras.layers.Dense((sample_len - 1) * 6)
        # x = dense(x)
        return x
