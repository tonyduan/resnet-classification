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
    argparser.add_argument("--score", type=str, default="nll")
    argparser.add_argument("--adv", action="store_true")
    argparser.add_argument("--inspect", action="store_true")
    args = argparser.parse_args()

    df = defaultdict(list)

    experiment_names = filter(lambda s: os.path.isdir(f"{args.dir}/{s}"), os.listdir(f"{args.dir}"))

    for experiment_name in experiment_names:

        experiment_df = pd.read_csv(f"{args.dir}/{experiment_name}/results.csv")

        for split, adv, interval, key in itertools.product(("train", "test"), ("", "_adv") if args.adv else ("",),

            if interval == "":
                col = f"{split}{adv}_{key}"
            else:
                col = f"{split}{adv}_{interval}_{key}"
            df["interval"].extend([interval] * len(experiment_df))
            df["split"].extend([split] * len(experiment_df))
            df["key"].extend([key] * len(experiment_df))
            df["value"].extend(experiment_df[col])
            df["epoch"].extend(np.arange(len(experiment_df)) + 1)

    df = pd.DataFrame(df)
    df.to_csv(f"{args.dir}/loss_curves.csv")

    if args.inspect:
        breakpoint()

    sns.set_style("white", {"font.family": "Times New Roman"})
    grid = sns.relplot(data=df, kind="line", x="epoch", y="value", row="key", col="experiment",
                       hue="split", hue_order=("train", "test"),
                       style="adv", dashes={"none": "", "pgd": (2, 4)},
                       size="interval", sizes=(1, 1),
                       palette=sns.color_palette("gray", 2),
                       row_order=("nll", "acc", "toplabel_ece", "consistency_toplabel_ece"),
                       col_order=sorted(set(df["experiment"])),
                       aspect=1.6, height=1.2, facet_kws={"sharex": True, "sharey": "row", "legend_out": True})

    if args.dir.startswith("cifar100"):
        ylims = {
            "nll": 5,
            "acc": 1.05,
            "ece": 0.3,
            "consistency_ece": 0.05,
        }
    elif args.dir.startswith("cifar10"):
        ylims = {
            "nll": 2.5,
            "acc": 1.05,
            "ece": 0.4 if args.adv else 0.2,
            "consistency_ece": 0.025,
        }
    else:
        raise ValueError

    grid.set_xlabels("Epoch")
    grid.set_titles("{col_name}")
    grid.axes[0][0].set_ylabel("NLL")
    grid.axes[0][0].set_ylim((0, ylims["nll"]))
    grid.axes[1][0].set_ylabel("Accuracy")
    grid.axes[1][0].set_ylim((0, ylims["acc"]))
    grid.axes[2][0].set_ylabel("ECE")
    grid.axes[2][0].set_ylim((0, ylims["ece"]))
    grid.axes[3][0].set_ylabel("Consistency ECE")
    grid.axes[3][0].set_ylim((0, ylims["consistency_ece"]))
    grid.tight_layout()
    grid.savefig(f"{args.dir}/loss_curves.pdf")
    plt.show()
