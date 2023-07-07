from keras.callbacks import EarlyStopping
import wandb
from wandb.keras import WandbCallback
from prosit_t.models import PrositSimpleIntensityPredictor
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance
from dlomix.constants import ALPHABET_UNMOD
from train_utils import get_example_data, get_proteometools_data
from prosit_t.optimizers.cyclic_lr import CyclicLR
import os

PROJECT_NAME = "transforming-prosit"
EPOCHS = 200

DEFAULT_CONFIG = {
    "learning_rate": 0.0002,
    "batch_size": 512,
    "embedding_output_dim": 64,
    "seq_length": 30,
    "len_fion": 6,
    "vocab_dict": ALPHABET_UNMOD,
    "dropout_rate": 0.2,
    "ff_dim": 32,
    "num_heads": 16,
    "transformer_dropout": 0.1,
    "dataset": "proteometools",
    "data_source": "/cmnfs/home/l.mamisashvili/transforming-prosit/notebooks/input_config.json",
    "fragmentation": "HCD",
    "mass_analyzer": "FTMS",
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


def get_callbacks(config):
    PATIENCE = 10
    callback_earlystopping = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1
    )
    cb_wandb = WandbCallback()
    cb_cyclic_lr = CyclicLR(base_lr=0.0000001, max_lr=0.001, step_size=8)
    callbacks = [callback_earlystopping, cb_wandb, cb_cyclic_lr]
    return callbacks


def train(config=None):
    with wandb.init(config=config, project=PROJECT_NAME) as run:
        config = wandb.config
        config = dict(wandb.config)

        if config["dataset"] == "example":
            train_dataset, val_dataset = get_example_data(config)
        else:
            assert "data_source" in config
            train_dataset, val_dataset = get_proteometools_data(config)

        model = get_model(config)
        callbacks = get_callbacks(config)
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=callbacks,
        )
        model.summary()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
