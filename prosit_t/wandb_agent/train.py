import tensorflow as tf
from prosit_t.wandb_agent.train_utils import train_generic
import os
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file", type=str, help="Config yaml path")
    parser.add_argument("cuda_idx", type=str, help="Index for the visible gpu")
    parser.add_argument("project_name", type=str, help="Wandb project name")
    args = parser.parse_args()
    config_file = args.config_file
    cuda_idx = args.cuda_idx
    project_name = args.project_name

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    with open(config_file, "r") as yaml_file:
        try:
            config_dict = yaml.safe_load(yaml_file)
        except yaml.YAMLError as e:
            print("Error parsing YAML:", e)
    if config_dict["dataset"] == "proteometools_dynamic_len":
        tf.config.run_functions_eagerly(True)

    train_generic(config_dict, project_name)


if __name__ == "__main__":
    main()
