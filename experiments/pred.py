import numpy as np
import pandas as pd
import torch
import argparse
import pandas as pd
from experiments.murenn_svdd import MurennExperiment
import yaml

from data import INVERSE_SNR_MAP, INVERSE_CLASS_MAP

def run(ckpt_path, config_path):
    # Load experiment config
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    experiment = MurennExperiment.load_from_checkpoint(ckpt_path, **cfg)
    experiment.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()
    run(args.ckpt_path, args.config_path)
