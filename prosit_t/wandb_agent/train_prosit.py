from dlomix.models import PrositIntensityPredictor
from dlomix.losses import masked_spectral_distance
from prosit_t.constants import ALPHABET_UNMOD
from prosit_t.wandb_agent.train_utils import train
import tensorflow as tf
import os

PROJECT_NAME = "transforming-prosit"
DEFAULT_CONFIG = {
    "learning_rate": 1e-3,
    "batch_size": 1024,
    "embedding_output_dim": 16,
    "seq_length": 30,
    "len_fion": 6,
    "vocab_dict": ALPHABET_UNMOD,
    "recurrent_layers_sizes": (256, 512),
    "regressor_layer_size": 512,
    "dataset": "proteometools_filtered_ftms",
    "data_source": {
        "train": "/cmnfs/proj/prosit/Transformer/all_unmod_train.parquet",
        "val": "/cmnfs/proj/prosit/Transformer/all_unmod_test.parquet",
    },
    "fragmentation": "HCD",
    "early_stopping": {
        "patience": 30,
        "min_delta": 0.0001,
    },
    "cyclic_lr": {
        "base_lr": 0.00001,
        "gamma": 0.95,
        "max_lr": 0.0002,
        "mode": "triangular",
        "step_size": 4,
    },
    "epochs": 200,
}


def get_model(config):
    model = PrositIntensityPredictor(
        seq_length=config["seq_length"],
        embedding_output_dim=config["embedding_output_dim"],
        recurrent_layers_sizes=config["recurrent_layers_sizes"],
        vocab_dict=config["vocab_dict"],
    )
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    train(DEFAULT_CONFIG, get_model)


if __name__ == "__main__":
    main()
