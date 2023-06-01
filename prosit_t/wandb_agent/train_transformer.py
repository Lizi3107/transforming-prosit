import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from prosit_t.models import PrositTransformerIntensityPredictor
from dlomix.losses import masked_spectral_distance, masked_pearson_correlation_distance
from dlomix.data import IntensityDataset
from dlomix.constants import ALPHABET_UNMOD

TRAIN_DATAPATH = "https://raw.githubusercontent.com/wilhelm-lab/dlomix-resources/main/example_datasets/Intensity/proteomeTools_train_val.csv"
PROJECT_NAME = "transforming-prosit"
EPOCHS = 30
DEFAULT_CONFIG = {
    "name": "",
    "learning_rate": 0.0001,
    "batch_size": 64,
    "embedding_output_dim": 512,
    "intermediate_dim": 512,
    "transformer_num_heads": 4,
    "mh_num_heads": 4,
    "key_dim": 4,
    "seq_length": 30,
    "len_fion": 6,
    "vocab_dict": ALPHABET_UNMOD,
    "dropout_rate": 0.2,
    "latent_dropout_rate": 0.1,
    "layer_norm_epsilon": 1e-5,
    "regressor_layer_size": 512,
}


def get_model(config):
    model = PrositTransformerIntensityPredictor()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=masked_spectral_distance,
        metrics=[masked_pearson_correlation_distance],
    )

    return model


def get_callbacks(config):
    cb_wandb = WandbCallback()
    callbacks = [cb_wandb]
    return callbacks


def get_data(run, config):
    BATCH_SIZE = config["batch_size"]
    int_data = IntensityDataset(
        data_source=TRAIN_DATAPATH,
        seq_length=config["seq_length"],
        collision_energy_col="collision_energy",
        batch_size=BATCH_SIZE,
        val_ratio=0.2,
        test=False,
    )
    return int_data.train_data, int_data.val_data


def train(config=None):
    with wandb.init(config=config, project=PROJECT_NAME) as run:
        config = wandb.config
        config = dict(wandb.config)

        train_dataset, val_dataset = get_data(run, config)
        model = get_model(config)
        callbacks = get_callbacks(config)
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=callbacks,
        )


def main():
    train(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
