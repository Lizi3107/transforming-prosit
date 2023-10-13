from dlomix.losses import masked_pearson_correlation_distance
import numpy as np
import os
from prosit_t.wandb_agent.train_utils import get_proteometools_data
from tqdm import tqdm
import wandb
from prosit_t.eval import prosit_transformer_eval
from dlomix.models import PrositIntensityPredictor


def compute_pc_distance(model, val_data, num_batches, batch_size):
    pc_dist_vals = []
    for sample in tqdm(val_data.take(num_batches)):
        x, y = sample
        predictions = model.predict(x, batch_size=batch_size)
        pc = masked_pearson_correlation_distance(y, predictions)
        pc_dist_vals.append(pc)
    return pc_dist_vals


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    project_name = "transforming-prosit-first-pool"
    run = wandb.init(project=project_name)
    _, val_data = get_proteometools_data(data_config)

    baseline_path = (
        "prosit-compms/transforming-prosit-first-pool/model-classic-star-15:v45"
    )
    baseline = PrositIntensityPredictor(
        seq_length=30, embedding_output_dim=16, recurrent_layers_sizes=(256, 512)
    )
    baseline = prosit_transformer_eval.load_model(run, baseline_path, baseline)
    pc_dist_vals = compute_pc_distance(
        baseline, val_data, len(val_data), data_config["batch_size"]
    )
    np.save("baseline_pearson.npy", np.array(pc_dist_vals))


data_config = {
    "dataset": "proteometools",
    "data_source": {
        "train": "/cmnfs/proj/prosit/Transformer/first_pool_train.parquet",
        "val": "/cmnfs/proj/prosit/Transformer/first_pool_test.parquet",
    },
    "fragmentation": "HCD",
    "batch_size": 1,
    "seq_length": 30,
}

if __name__ == "__main__":
    main()
