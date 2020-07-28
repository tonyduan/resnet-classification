import numpy as np
import pandas as pd
import itertools
import pickle
import seaborn as sns
import matplotlib as mpl
import os
from argparse import ArgumentParser
from collections import defaultdict
from matplotlib import pyplot as plt
from dfply import *


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dir", type=str, default="ckpts")
    argparser.add_argument("--score", type=str, default="nll")
    args = argparser.parse_args()

    df = defaultdict(list)

    experiment_names = filter(lambda s: os.path.isdir(f"{args.dir}/{s}"), os.listdir(f"{args.dir}"))

    for experiment_name in experiment_names:

        experiment_df = pd.read_csv(f"{args.dir}/{experiment_name}/results.csv")
        for split, interval, key in itertools.product(("train", "val", "test"),
                                                        ("", "lower", "upper"),
                                                        ("nll", "acc", "toplabel_ece", "consistency_toplabel_ece")):
            if interval == "":
                col = f"{split}_{key}"
            else:
                col = f"{split}_{interval}_{key}"

            df["experiment"].extend([experiment_name] * len(experiment_df))
            df["interval"].extend([interval] * len(experiment_df))
            df["split"].extend([split] * len(experiment_df))
            df["key"].extend([key] * len(experiment_df))
            df["value"].extend(experiment_df[col])
            df["epoch"].extend(np.arange(len(experiment_df)) + 1)

#        experiment_args = pickle.load(open(f"{args.dir}/{experiment_name}/args.pkl", "rb"))
#
#        for split, suffix in itertools.product(("train", "test"), ("", "acc", "res", "rel", "unc")):
#
#            if suffix == "":
#                vector = np.load(f"{args.dir}/{experiment_name}/{split}_{args.score}.npy")
#                df["score"].extend(vector)
#            elif suffix == "acc":
#                vector = np.load(f"{args.dir}/{experiment_name}/{split}_{suffix}.npy")
#                df["score"].extend(vector)
#            else:
#                vector = np.load(f"{args.dir}/{experiment_name}/{split}_{args.score}_{suffix}.npy")
#                df["score"].extend(vector)
#
#            suffix = "score" if suffix == "" else suffix
#            df["experiment"].extend([experiment_name] * len(vector))
#            df["measure"].extend([suffix] * len(vector))
#            df["split"].extend([split] * len(vector))
#            df["epoch"].extend(np.arange(len(vector)))
#            #df["wd"].extend([float(experiment_args.weight_decay)] * len(vector))
#

    sns.set_style("white", {"font.family": "Times New Roman"})

#    mapping = {"cifar100_pgd": "PGD", "cifar100_pgd_mixup": "PGD with mixup"}
#    df = df >> mutate(experiment = df["experiment"].map(mapping))
#    grid = sns.relplot(data=df, kind="line", x="epoch", y="score", row="measure", col="experiment", hue="split",
#                       hue_order=("train", "test"),
#                       palette=sns.color_palette("gray", 2), row_order=("score", "res", "rel", "acc"),
#                       aspect=1.6, height=1.2, facet_kws={"sharex": True, "sharey": "row", "legend_out": True})
    grid = sns.relplot(data=df, kind="line", x="epoch", y="value", row="key", col="experiment", hue="split",
                       hue_order=("train", "test",), style="interval",
                       palette=sns.color_palette("gray", 2), #row_order=("score", "res", "rel", "acc"),
                       aspect=1.6, height=1.2, facet_kws={"sharex": True, "sharey": "row", "legend_out": True})

    plt.show()

    grid.set_xlabels("Epoch")
    grid.set_titles("{col_name}")
#    grid.set_titles("{col_name}")
    grid.axes[0][0].set_ylabel("NLL")
    grid.axes[0][0].set_ylim((0, 5))
    grid.axes[1][0].set_ylabel("Resolution")
    grid.axes[1][0].set_ylim((0, 5))
    grid.axes[2][0].set_ylabel("Reliability")
    grid.axes[2][0].set_ylim((0, 0.6))
    grid.axes[3][0].set_ylabel("Accuracy")
    grid.axes[3][0].set_ylim((0, 1))
    grid.tight_layout()
    grid.savefig(f"{args.dir}/loss_curves.pdf")
    plt.show()

