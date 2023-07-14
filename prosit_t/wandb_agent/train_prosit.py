import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from dlomix.models import PrositIntensityPredictor
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance
from dlomix.constants import ALPHABET_UNMOD
from prosit_t.wandb_agent.train_utils import get_example_data, get_proteometools_data, train
from prosit_t.optimizers.cyclic_lr import CyclicLR

PROJECT_NAME = "transforming-prosit"
DEFAULT_CONFIG = {
    "learning_rate": 0.0001,
    "batch_size": 64,
    "embedding_output_dim": 16,
    "seq_length": 30,
    "len_fion": 6,
    "vocab_dict": ALPHABET_UNMOD,
    "recurrent_layers_sizes": (256, 512),
    "regressor_layer_size": 512,
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
}


def get_model(config):
    model = PrositIntensityPredictor(
        seq_length=config["seq_length"],
        embedding_output_dim=config["embedding_output_dim"],
        recurrent_layers_sizes=config["recurrent_layers_sizes"],
    )
    model.compile(
        optimizer="adam",
        loss=masked_spectral_distance,
        metrics=[masked_pearson_correlation_distance],
    )

    return model


def get_callbacks(config):
    cb_cyclic_lr = CyclicLR(base_lr=0.0000001, max_lr=0.001, step_size=8)
    cb_wandb = WandbCallback()
    callbacks = [cb_wandb, cb_cyclic_lr]
    return callbacks


def main():
    train(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
