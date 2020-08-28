import numpy as np
import pandas as pd
import itertools
import os
import pickle
import seaborn as sns
import matplotlib as mpl
from argparse import ArgumentParser
from collections import defaultdict
from matplotlib import pyplot as plt
from dfply import *


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dir", type=str, default="ckpts")
    argparser.add_argument("--inspect", action="store_true")
    args = argparser.parse_args()

    df = defaultdict(list)

    experiment_names = filter(lambda s: os.path.isdir(f"{args.dir}/{s}"), os.listdir(f"{args.dir}"))

    for experiment_name in experiment_names:

        experiment_df = pd.read_csv(f"{args.dir}/{experiment_name}/eval_results.csv")
        for k, v in experiment_df.items():
            df[k].extend(list(v))
        df["experiment"].extend(["_".join(experiment_name.split("_")[1:])] * len(v))

    df = pd.DataFrame(df)

    df = df >> mask(X.experiment.str.endswith("0.0"))
    df >> arrange(X.lower_acc, ascending=False)
    best_experiment = (df >> arrange(X.lower_acc, ascending=False))["experiment"].iloc[0]
    df = df >> mask(X.experiment == best_experiment)

    acc_df = df >> select(X.experiment, X.temperature_label, X.temperature, X.acc, X.lower_acc, X.upper_acc, X.pgd_acc, X.pgd_lower_acc, X.pgd_upper_acc) \
                >> arrange(X.temperature_label, X.experiment)
    ece_df = df >> select(X.experiment, X.temperature_label,  X.toplabel_ece, X.lower_toplabel_ece, X.upper_toplabel_ece, X.pgd_toplabel_ece, X.pgd_lower_toplabel_ece, X.pgd_upper_toplabel_ece) \
                >> arrange(X.temperature_label, X.experiment)
    cece_df = df >> select(X.experiment, X.temperature_label, X.lower_consistency_toplabel_ece, X.upper_consistency_toplabel_ece) \
                 >> arrange(X.temperature_label, X.experiment)
    print(acc_df.round(decimals=3))
    print(ece_df.round(decimals=3))
    print(cece_df.round(decimals=3))
    cece_df = df >> select(X.experiment, X.temperature_label, X.pgd_lower_consistency_toplabel_ece, X.pgd_upper_consistency_toplabel_ece) \
                 >> arrange(X.temperature_label, X.experiment)
    print(cece_df.round(decimals=3))

