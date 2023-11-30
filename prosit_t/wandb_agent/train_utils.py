from dlomix.data import IntensityDataset
import glob
import os
from pathlib import Path
import itertools
import json
import wandb
from wandb.keras import WandbCallback
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from prosit_t.optimizers.cyclic_lr import CyclicLR
import tensorflow as tf
from prosit_t.data.parquet_to_tfdataset import get_tfdatasets
from dlomix.losses import masked_spectral_distance
import importlib
from prosit_t.data.parquet_to_tfdataset_padded_filtered import (
    get_tfdatasets_padded_filtered,
)

DATA_DIR = "/cmnfs/proj/prosit/Transformer/"
META_DATA_DIR = "/cmnfs/proj/prosit/Transformer/Final_Meta_Data/"
TRAIN_DATAPATH = "https://raw.githubusercontent.com/wilhelm-lab/dlomix-resources/main/example_datasets/Intensity/proteomeTools_train_val.csv"
PROJECT_NAME = "transforming-prosit-first-pool"


def create_data_source_json(pool_keyword):
    meta_data_filepath = glob.glob(
        os.path.join(META_DATA_DIR, "*" + str(pool_keyword) + "*meta_data.parquet")
    )[0]
    annotation_dirs = [
        path
        for path in glob.glob(os.path.join(DATA_DIR, "*" + str(pool_keyword) + "*"))
        if os.path.isdir(path)
    ]
    annotations_filepaths = [
        glob.glob(os.path.join(d, "*.parquet")) for d in annotation_dirs
    ]
    annotations_filepaths = list(itertools.chain(*annotations_filepaths))
    annotations_names = [Path(f).stem for f in annotations_filepaths]
    input_data_dict = {
        "metadata": meta_data_filepath,
        "annotations": {
            pool_keyword: dict(zip(annotations_names, annotations_filepaths))
        },
        "parameters": {"target_column_key": "intensities_raw"},
    }
    with open("input_config.json", "w") as fp:
        json.dump(input_data_dict, fp)


def get_example_data(config):
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


def get_proteometools_data(config):
    train_path, val_path = config["data_source"].values()
    BATCH_SIZE = config["batch_size"]
    int_data_train = IntensityDataset(
        data_source=train_path,
        seq_length=30,
        batch_size=BATCH_SIZE,
        val_ratio=0,
        precursor_charge_col="precursor_charge_onehot",
        sequence_col="modified_sequence",
        collision_energy_col="collision_energy_aligned_normed",
        intensities_col="intensities_raw",
        parser="proforma",
        test=False,
    )
    int_data_val = IntensityDataset(
        data_source=val_path,
        seq_length=30,
        batch_size=BATCH_SIZE,
        val_ratio=0,
        precursor_charge_col="precursor_charge_onehot",
        sequence_col="modified_sequence",
        collision_energy_col="collision_energy_aligned_normed",
        intensities_col="intensities_raw",
        parser="proforma",
        test=False,
    )

    return int_data_train.train_data, int_data_val.train_data


def get_proteometools_data_variable_len(config):
    train_data, val_data = get_tfdatasets(config)
    return train_data, val_data


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def get_callbacks(config):
    cb_wandb = WandbCallback()

    callback_earlystopping = EarlyStopping(
        monitor="val_loss",
        patience=config["early_stopping"]["patience"],
        min_delta=config["early_stopping"]["min_delta"],
        restore_best_weights=True,
        verbose=1,
    )

    callbacks = [cb_wandb, callback_earlystopping]
    if "reduce_lr" in config:
        callback_reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=config["reduce_lr"]["factor"],
            patience=config["reduce_lr"]["patience"],
        )
        callbacks.append(callback_reduce_lr)
    if "cyclic_lr" in config:
        cb_cyclic_lr = CyclicLR(
            base_lr=config["cyclic_lr"]["base_lr"],
            max_lr=config["cyclic_lr"]["max_lr"],
            step_size=config["cyclic_lr"]["step_size"],
            gamma=config["cyclic_lr"]["gamma"],
            mode=config["cyclic_lr"]["mode"],
        )
        callbacks.append(cb_cyclic_lr)
    if "lr_scheduler" in config:
        cb_lr_scheduler = LearningRateScheduler(scheduler)
        callbacks.append(cb_lr_scheduler)
    return callbacks


def get_model(config):
    package = importlib.import_module(config["package_name"])
    model_class = getattr(package, config["model_class"])
    model = model_class(**config)
    if "cyclic_lr" in config:
        optimizer = "adam"
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss=masked_spectral_distance,
    )

    return model


def train_generic(config, project_name):
    with wandb.init(config=config, project=project_name) as _:
        config = wandb.config
        config = dict(wandb.config)

        if config["dataset"] == "example":
            train_dataset, val_dataset = get_example_data(config)
        elif config["dataset"] == "proteometools":
            assert "data_source" in config
            assert "train" in config["data_source"]
            assert "val" in config["data_source"]
            train_dataset, val_dataset = get_proteometools_data(config)
        elif config["dataset"] == "proteometools_dynamic_len":
            train_dataset, val_dataset = get_proteometools_data_variable_len(config)
        elif config["dataset"] == "proteometools_filtered_ftms":
            train_dataset, val_dataset = get_tfdatasets_padded_filtered(config)
        model = get_model(config)
        callbacks = get_callbacks(config)
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config["epochs"],
            callbacks=callbacks,
        )
        model.summary()


def train(config, get_model):
    with wandb.init(config=config, project=PROJECT_NAME) as _:
        config = wandb.config
        config = dict(wandb.config)

        if config["dataset"] == "example":
            train_dataset, val_dataset = get_example_data(config)
        elif config["dataset"] == "proteometools":
            assert "data_source" in config
            assert "train" in config["data_source"]
            assert "val" in config["data_source"]
            train_dataset, val_dataset = get_proteometools_data(config)
        elif config["dataset"] == "proteometools_dynamic_len":
            train_dataset, val_dataset = get_proteometools_data_variable_len(config)
        elif config["dataset"] == "proteometools_filtered_ftms":
            train_dataset, val_dataset = get_tfdatasets_padded_filtered(config)
        model = get_model(config)
        callbacks = get_callbacks(config)
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config["epochs"],
            callbacks=callbacks,
        )
        model.summary()
