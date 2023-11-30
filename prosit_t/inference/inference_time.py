import time
import pandas as pd
import matplotlib.pyplot as plt

WARMUP_EPOCHS = 10


def record_time(model, test_batch, repetitions=300):
    for _ in range(WARMUP_EPOCHS):
        _ = model.predict(test_batch)
    recordings = []
    for i in range(repetitions):
        start_time = time.perf_counter()
        model.predict(test_batch)
        end_time = time.perf_counter()
        recordings.append(end_time - start_time)
    return sum(recordings) / repetitions


def get_model_times(models_dict, test_batch):
    df = pd.DataFrame(columns=["model", "time"])
    for name, model in models_dict.items():
        inference_time = record_time(model, test_batch)
        df = df.append({"model": name, "time": inference_time}, ignore_index=True)
    return df


def plot_inference_time_comparison(df, baseline_name):
    df = df.sort_values(by="time")
    times = df["time"].tolist()
    model_names = df["model"].tolist()
    baseline_time = df[df.model == baseline_name].time.iloc[0]
    percentages = (times / baseline_time) * 100
    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(
        model_names,
        percentages,
        color=df["color"].tolist(),
        width=0.4,
        align="center",
        linewidth=3.5,
    )
    for bar, time_ in zip(bars, times):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{time_:.3f} s",
            ha="center",
            va="bottom",
        )
    ax.set_yticklabels([f"{int(x)}%" for x in ax.get_yticks()])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("Models", fontsize="11")
    plt.ylabel(
        f"Inference Time Relative to {baseline_name} (Percentage)", fontsize="11"
    )
    plt.xticks(rotation=30, ha="center")
    plt.show()
