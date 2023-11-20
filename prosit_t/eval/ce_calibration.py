import tensorflow as tf
import numpy as np
import pandas as pd
from prosit_t.eval import prosit_transformer_eval
import matplotlib.pyplot as plt
import seaborn as sns


def merge_tuples(item1, item2):
    return (
        {
            "sequence": item1[0],
            "collision_energy": item1[1],
            "precursor_charge": item1[2],
        },
        item2,
    )


def get_ce_list_for_pred(min_ce, max_ce):
    return np.arange(min_ce, max_ce) / 100


def dataset_to_df(dataset):
    val_df = pd.DataFrame(prosit_transformer_eval.dataset_to_list(dataset))
    val_df = val_df.explode(
        ["sequence", "collision_energy", "precursor_charge", "target"]
    )
    val_df = val_df.rename(
        columns={"collision_energy": "collision_energy_aligned_normed"}
    )
    val_df = val_df.reset_index(drop=True)
    return val_df


def get_orig_ce_column(val_df, val_parquet_path):
    parquet_df = pd.read_parquet(val_parquet_path)
    val_df["orig_collision_energy"] = parquet_df["orig_collision_energy"]
    val_df["raw_file"] = parquet_df["raw_file"]
    return val_df


def df_to_dataset(df, batch_size):
    sequence_col = df["sequence"].tolist()
    collision_energy_col = df["ce_for_pred"].tolist()
    precursor_charge_col = df["precursor_charge"].tolist()
    target_col = df["target"].tolist()

    sequence_tensor = tf.constant(sequence_col, dtype=tf.string)
    ce_tensor = tf.constant(collision_energy_col, dtype=tf.float64)
    charge_tensor = tf.constant(precursor_charge_col, dtype=tf.float32)
    target_tensor = tf.constant(target_col, dtype=tf.float32)

    dataset = (
        tf.data.Dataset.from_tensor_slices(
            ((sequence_tensor, ce_tensor, charge_tensor), target_tensor)
        )
        .map(merge_tuples)
        .batch(batch_size)
    )
    return dataset


def get_ce_calibration_df(val_df, model, batch_size=1024, min_ce=15, max_ce=50):
    result_df = pd.DataFrame(columns=["orig_ce", "ce_for_pred", "SA"])
    grouped = val_df.groupby("orig_collision_energy")
    ce_for_pred = get_ce_list_for_pred(min_ce, max_ce)
    for orig_ce, group in grouped:
        for ce_var in ce_for_pred:
            group["ce_for_pred"] = ce_var
            group["ce_for_pred"] = group["ce_for_pred"].apply(lambda x: [x])
            group = group.reset_index(drop=True)

            dataset = df_to_dataset(group, batch_size)
            num_batches = len(dataset)
            loss = prosit_transformer_eval.compute_losses(
                model, dataset, num_batches, batch_size
            )
            SA = 1 - np.array(loss)
            median = np.median(SA)
            new_row = {"orig_ce": orig_ce, "ce_for_pred": ce_var, "SA": median}
            new_row_df = pd.DataFrame(new_row, index=[0])
            result_df = pd.concat([result_df, new_row_df])
            result_df = result_df.reset_index(drop=True)

    return result_df


def plot_ce_calibration(
    result_df,
    color_palette="Set1",
    figure_size=(10, 8),
    title="CE Calibration for the Model",
    xlabel="NCE",
    ylabel="Median Spectral Angle",
    legend_title="Original CE",
):
    lines = result_df.iloc[result_df.groupby("orig_ce")["SA"].idxmax()]
    unique_colors = sns.color_palette(
        color_palette, n_colors=result_df["orig_ce"].nunique()
    )
    lines["color"] = unique_colors
    plt.figure(figsize=figure_size)
    plt.ylim(0, 1.0)
    sns.scatterplot(
        data=result_df,
        x="ce_for_pred",
        y="SA",
        hue="orig_ce",
        palette=color_palette,
        s=15,
    )
    for i, v in lines.iterrows():
        plt.axvline(x=v["ce_for_pred"], ymin=0, ymax=v["SA"], color=v["color"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=legend_title)
