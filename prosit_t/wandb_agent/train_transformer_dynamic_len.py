from prosit_t.models.variable_seq_length_models import ProstTransformerDynamicLen
import tensorflow as tf
from dlomix.losses import masked_spectral_distance
from dlomix.constants import ALPHABET_UNMOD
from prosit_t.wandb_agent.train_utils import train
import os

DEFAULT_CONFIG = {
    "learning_rate": 0.0001,
    "batch_size": 1024,
    "embedding_output_dim": 64,
    "meta_embedding_dim": 256,
    "len_fion": 6,
    "vocab_dict": ALPHABET_UNMOD,
    "dropout_rate": 0,
    "ff_dim": 32,
    "num_heads": 16,
    "transformer_dropout": 0.1,
    "dataset": "proteometools_dynamic_len",
    "data_source": {
        "train": "/cmnfs/proj/prosit/Transformer/first_pool_train.parquet",
        "val": "/cmnfs/proj/prosit/Transformer/first_pool_test.parquet",
    },
    "fragmentation": "HCD",
    "early_stopping": {
        "patience": 8,
        "min_delta": 0.0001,
    },
    "epochs": 200,
    "reduce_lr": {"factor": 0.8, "patience": 4},
    "num_transformers": 6,
}


def get_model(config):
    model = ProstTransformerDynamicLen(**config)
    if "cyclic_lr" in config:
        optimizer = "adam"
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss=masked_spectral_distance,
    )

    return model


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.config.run_functions_eagerly(True)

    train(DEFAULT_CONFIG, get_model)


if __name__ == "__main__":
    main()
