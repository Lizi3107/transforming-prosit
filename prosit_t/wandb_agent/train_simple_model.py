from prosit_t.models import PrositSimpleIntensityPredictor
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance
from dlomix.constants import ALPHABET_UNMOD
from train_utils import train
import os

PROJECT_NAME = "transforming-prosit"

DEFAULT_CONFIG = {
    "learning_rate": 0.0002,
    "batch_size": 1024,
    "embedding_output_dim": 64,
    "seq_length": 30,
    "len_fion": 6,
    "vocab_dict": ALPHABET_UNMOD,
    "dropout_rate": 0.2,
    "ff_dim": 32,
    "num_heads": 64,
    "transformer_dropout": 0.1,
    "dataset": "proteometools",
    "data_source": "/cmnfs/home/l.mamisashvili/transforming-prosit/prosit_t/data/first_pool.json",
    "fragmentation": "HCD",
    # "mass_analyzer": "FTMS",
    "cyclic_lr": {
        "max_lr": 0.0002,
        "base_lr": 0.00001,
        "mode": "triangular",
        "gamma": 0.95,
        "step_size": 4,
    },
    "early_stopping": {
        "patience": 30,
        "min_delta": 0.0001,
    },
    "epochs": 500,
    "num_transformers": 2,
}


def get_model(config):
    model = PrositSimpleIntensityPredictor(**config)
    model.compile(
        optimizer="adam",
        loss=masked_spectral_distance,
        metrics=[masked_pearson_correlation_distance],
    )

    return model


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train(DEFAULT_CONFIG, get_model)


if __name__ == "__main__":
    main()
