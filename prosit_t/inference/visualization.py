from plotly.offline import iplot
from plotly import graph_objs as go
import numpy as np


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
