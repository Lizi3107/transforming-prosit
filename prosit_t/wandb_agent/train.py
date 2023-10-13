import tensorflow as tf
from prosit_t.wandb_agent.train_utils import train_generic
import os
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file", type=str, help="Config yaml path")
    parser.add_argument("cuda_idx", type=str, help="Index for the visible gpu")
    args = parser.parse_args()
    config_file = args.config_file
    cuda_idx = args.cuda_idx

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    with open(config_file, "r") as yaml_file:
        try:
            config_dict = yaml.safe_load(yaml_file)
        except yaml.YAMLError as e:
            print("Error parsing YAML:", e)

    train_generic(config_dict)


if __name__ == "__main__":
    main()
