from dlomix.data import IntensityDataset
from dlomix.data.feature_extractors import (
    ModificationGainFeature,
    ModificationLocationFeature,
    ModificationLossFeature,
)
import glob
import os
from pathlib import Path
import itertools
import json
import wandb

DATA_DIR = "/cmnfs/proj/prosit/Transformer/"
META_DATA_DIR = "/cmnfs/proj/prosit/Transformer/Final_Meta_Data/"
TRAIN_DATAPATH = "https://raw.githubusercontent.com/wilhelm-lab/dlomix-resources/main/example_datasets/Intensity/proteomeTools_train_val.csv"
PROJECT_NAME = "transforming-prosit"


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
    data_source = config["data_source"]
    BATCH_SIZE = config["batch_size"]
    SEQ_LENGTH = config["seq_length"]
    FRAGMENTATION = config["fragmentation"]
    metadata_filtering_criteria = {
        "peptide_length": f"<= {SEQ_LENGTH}",
        "precursor_charge": "<= 6",
        "fragmentation": f"== {FRAGMENTATION}",
    }
    if "mass_analyzer" in config:
        metadata_filtering_criteria["mass_analyzer"] = f"== {config['mass_analyzer']}"
    int_data = IntensityDataset(
        data_source=data_source,
        seq_length=SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        val_ratio=0.15,
        precursor_charge_col="precursor_charge_onehot",
        sequence_col="modified_sequence",
        collision_energy_col="collision_energy_aligned_normed",
        intensities_col="intensities_raw",
        features_to_extract=[
            ModificationLocationFeature(),
            ModificationLossFeature(),
            ModificationGainFeature(),
        ],
        parser="proforma",
        metadata_filtering_criteria=metadata_filtering_criteria,
    )
    return int_data.train_data, int_data.val_data


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
            epochs=config["epochs"],
            callbacks=callbacks,
        )
        model.summary()
