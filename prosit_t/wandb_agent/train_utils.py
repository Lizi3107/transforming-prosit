from prosit_t.data import IntensityDataset
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

DATA_DIR = "/cmnfs/proj/prosit/Transformer/"
META_DATA_DIR = "/cmnfs/proj/prosit/Transformer/Final_Meta_Data/"
TRAIN_DATAPATH = "https://raw.githubusercontent.com/wilhelm-lab/dlomix-resources/main/example_datasets/Intensity/proteomeTools_train_val.csv"


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
    MASS_ANALYZER = config["mass_analyzer"]
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
        sequence_filtering_criteria={
            "max_peptide_length": SEQ_LENGTH,
            "max_precursor_charge": 6,
        },
        fragmentation_filter=FRAGMENTATION,
        mass_analyzer_filter=MASS_ANALYZER,
    )
    return int_data.train_data, int_data.val_data
