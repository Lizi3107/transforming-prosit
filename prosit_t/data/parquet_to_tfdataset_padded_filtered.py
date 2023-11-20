import tensorflow as tf
import numpy as np
import pandas as pd


X_COLUMNS = ["sequence", "precursor_charge", "collision_energy"]
MOD_ENCODING = ["M[UNIMOD:35]", "C[UNIMOD:4]"]
MAX_SEQUENCE_LEN = 30


def merge_tuples(item1, item2):
    return (
        {
            "sequence": item1[0],
            "collision_energy": item1[1],
            "precursor_charge": item1[2],
        },
        item2,
    )


def create_input_df(parquet_df):
    df = pd.DataFrame(columns=X_COLUMNS)
    df["sequence"] = (
        parquet_df["modified_sequence"]
        .str.replace(MOD_ENCODING[0], "m", regex=False)
        .str.replace(MOD_ENCODING[1], "C", regex=False)
    )
    df["precursor_charge"] = parquet_df["precursor_charge_onehot"]

    df["collision_energy"] = parquet_df["collision_energy_aligned_normed"]
    df["target"] = parquet_df["intensities_raw"]
    return df


def pad_sequences(parquet_df, seq_len):
    parquet_df["sequence"] = parquet_df.sequence.apply(
        lambda x: np.pad(
            np.array(list(x)),
            (0, seq_len - len(x)),
            "constant",
            constant_values=b"",
        )
    )
    return parquet_df


def expand_ce_dim(parquet_df):
    parquet_df.loc[:, "collision_energy"] = parquet_df["collision_energy"].apply(
        lambda x: [x]
    )
    return parquet_df


def df_to_tfdataset(df, batch_size):
    sequence_col = df["sequence"].tolist()
    collision_energy_col = df["collision_energy"].tolist()
    precursor_charge_col = df["precursor_charge"].tolist()
    target_col = df["target"].tolist()

    sequence_tensor = tf.constant(sequence_col, dtype=tf.string)
    ce_tensor = tf.constant(collision_energy_col, dtype=tf.float32)
    charge_tensor = tf.constant(precursor_charge_col, dtype=tf.float32)
    target_tensor = tf.constant(target_col, dtype=tf.float64)

    dataset = (
        tf.data.Dataset.from_tensor_slices(
            ((sequence_tensor, ce_tensor, charge_tensor), target_tensor)
        )
        .map(merge_tuples)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset


def get_tfdatasets_padded_filtered(config):
    train_df = pd.read_parquet(config["data_source"]["train"])
    val_df = pd.read_parquet(config["data_source"]["val"])

    train_df = train_df[train_df.mass_analyzer == "FTMS"]
    val_df = val_df[val_df.mass_analyzer == "FTMS"]

    train_df = create_input_df(train_df)
    val_df = create_input_df(val_df)

    train_df = pad_sequences(train_df, config["seq_length"])
    val_df = pad_sequences(val_df, config["seq_length"])

    train_df = expand_ce_dim(train_df)
    val_df = expand_ce_dim(val_df)

    train_ds = df_to_tfdataset(train_df, config["batch_size"])
    val_ds = df_to_tfdataset(val_df, config["batch_size"])

    return train_ds, val_ds
