from plotly.offline import iplot
from plotly import graph_objs as go
import numpy as np
import matplotlib.pyplot as plt


def compare_spectra(y_true, y_pred, labels, barmode="group"):
    mask = np.not_equal(y_true, -1).astype(np.float64)
    masked_y_pred = y_pred * mask
    masked_y_pred[masked_y_pred < 0] = 0
    masked_y_pred = masked_y_pred / masked_y_pred.max()

    fig = {
        "data": [
            go.Bar(x=np.arange(len(y_true)), y=y_true * mask, name="Target"),
            go.Bar(x=np.arange(len(y_true)), y=masked_y_pred, name="Prediction"),
        ],
        "layout": go.Layout(
            barmode=barmode,
        ),
    }
    iplot(fig, show_link=False)


def get_annotations(y_true):
    annotations = []
    x_values = np.arange(len(y_true))
    num_b = 0
    num_y = 0
    for i in range(0, len(x_values), 1):
        if i % 3 == 0:
            if (i // 3) % 2:
                current_ion = "b"
                num_b += 1
            else:
                current_ion = "y"
                num_y += 1
        idx = num_b if current_ion == "b" else num_y
        annotations.append(f"{current_ion}{idx}(+{(i % 3) + 1})")
    return annotations


def compare_spectra_annotated(y_true, y_pred):
    x_values = np.arange(len(y_true))
    annotations = get_annotations(y_true)

    mask = np.not_equal(y_true, -1).astype(np.float64)
    masked_y_pred = y_pred * mask
    masked_y_pred[masked_y_pred < 0] = 0
    masked_y_pred = masked_y_pred / masked_y_pred.max()

    pred_values = masked_y_pred
    target_values = y_true * mask
    condition = (pred_values != 0) | (target_values != 0)
    height_difference = pred_values - target_values

    negative_diff_mask = (height_difference < 0).numpy().astype(np.float64)
    positive_diff_mask = (height_difference > 0).numpy().astype(np.float64)
    overlap = np.minimum(pred_values, target_values)

    mask_b_ions = np.tile([1, 1, 1, 0, 0, 0], len(overlap) // 6)
    mask_y_ions = np.tile([0, 0, 0, 1, 1, 1], len(overlap) // 6)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x_values,
        overlap * mask_b_ions,
        color="#a94442",
        label="Matched y-ion",
        width=0.5,
    )
    ax.bar(
        x_values,
        overlap * mask_y_ions,
        color="#425ba9",
        label="Matched b-ion",
        width=0.5,
    )
    ax.bar(
        x_values,
        -(height_difference * negative_diff_mask),
        color="#659732",
        bottom=pred_values,
        label="Synthetic",
        width=0.5,
    )
    ax.bar(
        x_values,
        height_difference * positive_diff_mask,
        color="#dfd42d",
        bottom=target_values,
        label="Predicted",
        width=0.5,
    )

    for i, val in enumerate(annotations):
        if target_values[i] != 0:
            txt_y = max(pred_values[i], target_values[i]) + 0.01
            ha = "center" if txt_y < 1 else "left"
            va = "bottom" if txt_y < 1 else "top"
            ax.text(
                x_values[i],
                max(pred_values[i], target_values[i]) + 0.01,
                annotations[i],
                ha=ha,
                va=va,
                color="black",
                rotation=90,
            )

    ax.set_xlabel("X Values")
    ax.set_ylabel("Y Values")
    ax.legend()
    ax.set_xlim(0, np.where(condition)[0][-1])
    plt.show()


def compare_multiple_spectra(
    y_true, y_preds, pred_labels, losses, barmode="group", bar_width=0.3
):
    mask = np.not_equal(y_true, -1).astype(np.float64)
    masked_y_preds = []
    for y_pred in y_preds:
        masked_y_pred = y_pred * mask
        masked_y_pred[masked_y_pred < 0] = 0
        masked_y_pred = masked_y_pred / masked_y_pred.max()
        masked_y_preds.append(masked_y_pred)

    bars = [
        go.Bar(
            x=np.arange(len(y_true)),
            y=y_true * mask,
            name="Target",
            width=bar_width,
            marker_color="#e48b4e",
        )
    ]
    for idx, e in enumerate(masked_y_preds):
        bars.append(
            go.Bar(
                x=np.arange(len(y_true)),
                y=e,
                name=f"{pred_labels[idx]}, SD - {losses[idx]:.3f}",
                width=bar_width,
            ),
        )
    fig = {
        "data": bars,
        "layout": go.Layout(
            barmode=barmode,
        ),
    }
    iplot(fig, show_link=False)
