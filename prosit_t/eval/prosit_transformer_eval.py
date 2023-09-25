import pandas as pd
from dlomix.losses import masked_spectral_distance
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


def dataset_to_list(tf_dataset):
    data_list = []
    for element in tf_dataset.as_numpy_iterator():
        x, y = element
        data_dict = {
            "sequence": x["sequence"],
            "collision_energy": x["collision_energy"],
            "precursor_charge": x["precursor_charge"],
            "target": y,
        }
        data_list.append(data_dict)
    return data_list


def load_model(run, artifact_path, model_obj):
    model_artifact = run.use_artifact(artifact_path)
    model_dir = model_artifact.download()
    model_obj.load_weights(model_dir)
    return model_obj


def compute_losses(model, val_data, num_batches, batch_size):
    losses = []
    for sample in val_data.take(num_batches):
        x, y = sample
        predictions = model.predict(x, batch_size=batch_size)
        loss = masked_spectral_distance(y, predictions)
        losses.extend(loss.numpy())
    return losses


def process_df(df):
    df = df.explode(["sequence", "collision_energy", "precursor_charge", "target"])
    df["sequence_length"] = df["sequence"].apply(
        lambda x: len([item for item in x if item != b""])
    )
    df["precursor_charge_int"] = df["precursor_charge"].apply(
        lambda x: int(np.where(x == 1)[0]) + 1
    )
    df["collision_energy"] = df["collision_energy"].apply(lambda x: x.squeeze())
    df["collision_energy_range"] = pd.cut(
        df["collision_energy"],
        bins=[0.2, 0.25, 0.3, 0.35, 0.4],
        labels=["< 0.25", "0.25-0.3", "0.3-0.35", "> 0.35"],
    )
    return df


def violin_plot_per_feature_val(
    df,
    loss_columns,
    feature_column,
    line_colors=["blue", "orange"],
    sides=["negative", "positive"],
    violinmode="overlay",
    title="My Plotly Figure Title",
    xaxis_title="x-axis title",
    yaxis_title="y-axis title",
):
    fig = go.Figure()
    for idx, loss_col in enumerate(loss_columns):
        name = loss_col.split("_")[0]
        fig.add_trace(
            go.Violin(
                x=df[feature_column],
                y=df[loss_col],
                legendgroup=name,
                scalegroup=name,
                name=name,
                side=sides[idx],
                line_color=line_colors[idx],
                points=False,
            )
        )
    fig.update_traces(meanline_visible=True)
    fig.update_layout(
        violingap=0,
        violinmode=violinmode,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    return fig


def histogram_per_feature_val(
    df,
    loss_column,
    feature_column,
    rows_n,
    cols_n,
    title="Histograms Title",
    xaxis_title="x-axis title",
    yaxis_title="y-axis title",
):
    fig = make_subplots(
        rows=rows_n,
        cols=cols_n,
        shared_yaxes=True,
        subplot_titles=list(map(str, df[feature_column].unique())),
    )
    row, col = 1, 1
    for i, group in enumerate(df[feature_column].unique()):
        subset = df[df[feature_column] == group]
        trace = go.Histogram(x=subset[loss_column], name=str(group))
        fig.add_trace(trace, row=row, col=col)
        if (i + 1) % cols_n == 0:
            row += 1
            col = 1
        else:
            col += 1

    fig.update_layout(
        title_text=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=False,
    )
    return fig