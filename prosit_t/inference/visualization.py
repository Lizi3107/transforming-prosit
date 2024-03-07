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


def get_annotations(y_true):
    annotations = []
    colors = []
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
        colors.append("#425ba9" if current_ion == "b" else "#a94442")
    return annotations, colors


def compare_spectra_annotated(y_true, y_pred, loss, model_name):
    sa = 1 - loss
    x_values = np.arange(len(y_true))
    annotations, colors = get_annotations(y_true)
    mask = np.logical_and(np.not_equal(y_true, -1), np.greater(y_true, 0.05)).astype(
        np.float64
    )
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
            ha = "left" if txt_y > 1 else "center"
            va = "bottom" if txt_y < 1 else "top"
            x_coordinates = x_values[i] + 0.5 if ha == "left" else x_values[i]
            ax.text(
                x_coordinates,
                max(pred_values[i], target_values[i]) + 0.01,
                annotations[i],
                ha=ha,
                va=va,
                color=colors[i],
                rotation=90,
            )
    ax.set_title(f"{model_name}: SA {sa:.3f}", pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Peak Index")
    ax.set_ylabel("Relative Peak Intensity")
    ax.legend(loc="upper center")
    ax.set_xlim(-1, np.where(condition)[0][-1] + 1)
    plt.show()


def compare_spectra_annotated_mirror(
    sample_x,
    y_true,
    model1_y_pred,
    model2_y_pred,
    model1_loss,
    model2_loss,
    model1_name,
    model2_name,
    legend_loc="upper left",
):
    x_values = np.arange(len(y_true))
    annotations, colors = get_annotations(y_true)
    model1_sa = 1 - model1_loss
    model2_sa = 1 - model2_loss
    (
        model1_pred_values,
        target_values,
        model1_target_extra,
        model1_pred_extra,
        model1_mask_b_ions,
        model1_mask_y_ions,
    ) = postprocess_peaks(y_true, model1_y_pred, model1_loss)
    (
        model2_pred_values,
        target_values,
        model2_target_extra,
        model2_pred_extra,
        model2_mask_b_ions,
        model2_mask_y_ions,
    ) = postprocess_peaks(y_true, model2_y_pred, model2_loss)
    condition1 = np.where((model1_pred_values != 0) | (target_values != 0))[0][-1]
    condition2 = np.where((model2_pred_values != 0) | (target_values != 0))[0][-1]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.bar(
        x_values,
        model1_mask_b_ions,
        color="#a94442",
        label="Matched y-ion",
        width=0.5,
    )
    ax.bar(
        x_values,
        model1_mask_y_ions,
        color="#425ba9",
        label="Matched b-ion",
        width=0.5,
    )
    ax.bar(
        x_values,
        -model1_target_extra,
        color="#659732",
        bottom=model1_pred_values,
        label="Synthetic",
        width=0.5,
    )
    ax.bar(
        x_values,
        model1_pred_extra,
        color="#dfd42d",
        bottom=target_values,
        label="Predicted",
        width=0.5,
    )

    ax.bar(
        x_values,
        -model2_mask_b_ions,
        color="#a94442",
        width=0.5,
    )
    ax.bar(
        x_values,
        -model2_mask_y_ions,
        color="#425ba9",
        width=0.5,
    )
    ax.bar(
        x_values,
        model2_target_extra,
        color="#659732",
        bottom=-model2_pred_values,
        width=0.5,
    )
    ax.bar(
        x_values,
        -model2_pred_extra,
        color="#dfd42d",
        bottom=-target_values,
        width=0.5,
    )
    ax.axhline(0, color="black", linewidth=1)

    for i, val in enumerate(annotations):
        if target_values[i] != 0:
            txt_y = max(model1_pred_values[i], target_values[i])
            ha = "left" if txt_y == 1 else "center"
            va = "bottom" if txt_y != 1 else "top"
            x_coordinates = x_values[i] + 0.5 if ha == "left" else x_values[i]
            ax.text(
                x_coordinates,
                txt_y + 0.05,
                annotations[i],
                ha=ha,
                va=va,
                color=colors[i],
                rotation=90,
            )
    for i, val in enumerate(annotations):
        if target_values[i] != 0:
            txt_y = max(model2_pred_values[i], target_values[i])
            ha = "left" if txt_y == 1 else "center"
            va = "top" if txt_y != 1 else "bottom"
            x_coordinates = x_values[i] if ha == "center" else x_values[i] + 0.5
            ax.text(
                x_coordinates,
                -(txt_y + 0.05),
                annotations[i],
                ha=ha,
                va=va,
                color=colors[i],
                rotation=90,
            )
    sequence_title =''.join(sample_x["sequence"].numpy().astype(str))
    charge_title = f"{(np.argmax(sample_x['precursor_charge']) + 1)}+"
    nce_title = round(float(sample_x["collision_energy"][0]),2)
    plt.figtext(
        0.1, 0.95, f"{sequence_title} {charge_title}, NCE: {nce_title}", ha="left", fontsize="12"
    )
    plt.figtext(
        0.5, 0.15, f"{model2_name}: SA {model2_sa:.3f}", ha="center", fontsize="11"
    )
    ax.set_title(
        f"{model1_name}: SA {model1_sa:.3f}",
        pad=20,
        fontsize="11"
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Peak Index")
    ax.set_ylabel("Relative Peak Intensity (%)")
    ax.legend(loc=legend_loc)
    ax.set_xlim(-1, max(condition1, condition2) + 1)
    ax.set_ylim(bottom=-1.1)
    plt.show()


def custom_formatter(x, pos):
    return f"{abs(int(x*100))}%"


def postprocess_peaks(y_true, y_pred, loss):
    mask = np.logical_and(np.not_equal(y_true, -1), np.greater(y_true, 0.05)).astype(
        np.float64
    )
    masked_y_pred = y_pred * mask
    masked_y_pred[masked_y_pred < 0] = 0
    masked_y_pred = masked_y_pred / masked_y_pred.max()

    pred_values = masked_y_pred
    target_values = y_true * mask

    height_difference = pred_values - target_values

    negative_diff_mask = (height_difference < 0).numpy().astype(np.float64)
    positive_diff_mask = (height_difference > 0).numpy().astype(np.float64)
    overlap = np.minimum(pred_values, target_values)
    mask_b_ions = np.tile([1, 1, 1, 0, 0, 0], len(overlap) // 6)
    mask_y_ions = np.tile([0, 0, 0, 1, 1, 1], len(overlap) // 6)
    return (
        pred_values,
        target_values,
        height_difference * negative_diff_mask,
        height_difference * positive_diff_mask,
        overlap * mask_b_ions,
        overlap * mask_y_ions,
    )
