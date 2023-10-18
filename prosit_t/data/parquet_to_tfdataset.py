import tensorflow as tf
import numpy as np
import pandas as pd

pd.set_option("mode.chained_assignment", None)

DATA_CONFIG = {
    "data_source": {
        "train": "/cmnfs/proj/prosit/Transformer/first_pool_train.parquet",
        "val": "/cmnfs/proj/prosit/Transformer/first_pool_test.parquet",
    }
}

X_COLUMNS = ["sequence", "precursor_charge", "collision_energy"]
MOD_ENCODING = ["M[UNIMOD:35]", "C[UNIMOD:4]"]


def concatenate_columns(row):
    return row.tolist()


def int_to_onehot(charge):
    onehot = np.zeros(6)
    onehot[charge - 1] = 1
    return onehot


def truncate_target(row):
    sequence = row["sequence"]
    target = row["intensities_raw"][: (len(sequence) - 1) * 6]
    return target


def ragged_to_dense(x, y):
    return x, y


def merge_tuples(item1, item2):
    return (
        {
            "sequence": item1[0],
            "precursor_charge": item2[0],
            "collision_energy": item2[1],
        },
        item1[-1],
    )


def create_input_df(parquet_df, encode_ox=True):
    df = pd.DataFrame(columns=X_COLUMNS)
    if encode_ox:
        df["sequence"] = (
            parquet_df["modified_sequence"]
            .str.replace(MOD_ENCODING[0], "m", regex=False)
            .str.replace(MOD_ENCODING[1], "C", regex=False)
        )
    else:
        df["sequence"] = (
            parquet_df["modified_sequence"]
            .str.replace(MOD_ENCODING[0], "M", regex=False)
            .str.replace(MOD_ENCODING[1], "C", regex=False)
        )
    df["precursor_charge"] = parquet_df["precursor_charge"].apply(int_to_onehot)

    df["collision_energy"] = parquet_df["collision_energy_aligned_normed"]
    df["intensities_raw"] = parquet_df["intensities_raw"]
    df["target"] = df.apply(truncate_target, axis=1)
    df = df.drop("intensities_raw", axis=1)
    return df


def process_input_df_columns(input_df):
    input_df.loc[:, "sequence"] = input_df["sequence"].apply(
        lambda x: np.array(list(x))
    )
    input_df.loc[:, "precursor_charge"] = input_df["precursor_charge"].apply(np.array)
    input_df.loc[:, "collision_energy"] = input_df["collision_energy"].apply(
        lambda x: [x]
    )
    return input_df


def df_to_ragged_tensors(df):
    sequence_col = df["sequence"].tolist()
    collision_energy_col = df["collision_energy"].tolist()
    precursor_charge_col = df["precursor_charge"].tolist()
    target_col = df["target"].tolist()

    sequence_ragged = tf.ragged.constant(sequence_col, dtype=tf.string)
    collision_energy_ragged = tf.ragged.constant(collision_energy_col, dtype=tf.float32)
    precursor_charge_ragged = tf.ragged.constant(precursor_charge_col, dtype=tf.float32)
    target_ragged = tf.ragged.constant(target_col, dtype=tf.float64)
    return (
        sequence_ragged,
        collision_energy_ragged,
        precursor_charge_ragged,
        target_ragged,
    )


def ragged_to_tfdataset(
    sequence_ragged,
    collision_energy_ragged,
    precursor_charge_ragged,
    target_ragged,
    batch_size=1024,
):
    dataset_seq_target = (
        tf.data.Dataset.from_tensor_slices(
            (
                sequence_ragged,
                target_ragged,
            )
        )
        .map(ragged_to_dense)
        .padded_batch(
            batch_size,
            padding_values=(tf.constant(""), tf.constant(-1, dtype=tf.float64)),
        )
        .unbatch()
    )

    dataset_meta = tf.data.Dataset.from_tensor_slices(
        (precursor_charge_ragged, collision_energy_ragged)
    ).map(ragged_to_dense)

    dataset = (
        tf.data.Dataset.zip(dataset_seq_target, dataset_meta)
        .map(merge_tuples)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def get_tfdatasets(batch_size, encode_ox=True):
    train_df = pd.read_parquet(DATA_CONFIG["data_source"]["train"])
    val_df = pd.read_parquet(DATA_CONFIG["data_source"]["val"])

    train_in_df = create_input_df(train_df, encode_ox)
    val_in_df = create_input_df(val_df, encode_ox)

    train_in_df = process_input_df_columns(train_in_df)
    val_in_df = process_input_df_columns(val_in_df)

    train_ragged = df_to_ragged_tensors(train_in_df)
    val_ragged = df_to_ragged_tensors(val_in_df)

    train_dataset = ragged_to_tfdataset(*train_ragged)
    val_dataset = ragged_to_tfdataset(*val_ragged, batch_size=batch_size)
    return train_dataset, val_dataset
