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
    args = argparser.parse_args()

    df = defaultdict(list)

    experiment_names = filter(lambda s: os.path.isdir(f"{args.dir}/{s}"), os.listdir(f"{args.dir}"))

    for experiment_name in experiment_names:

        experiment_args = pickle.load(open(f"{args.dir}/{experiment_name}/args.pkl", "rb"))

        for split, adv, suffix in itertools.product(("train", "test"), ("", "_adv"),
                                                    ("", "acc", "res", "rel", "unc", "toplabel_ece")):

            if suffix == "": # plug-in estimator
                vector = np.load(f"{args.dir}/{experiment_name}/{split}{adv}_{args.score}.npy")
                df["score"].extend(vector)
            elif suffix in ("res", "rel", "unc"):
                vector = np.load(f"{args.dir}/{experiment_name}/{split}{adv}_{args.score}_{suffix}.npy")
                df["score"].extend(vector)
            else:
                vector = np.load(f"{args.dir}/{experiment_name}/{split}{adv}_{suffix}.npy")
                df["score"].extend(vector)

            suffix = "score" if suffix == "" else suffix
            adv = "none" if adv == "" else "pgd"

            df["experiment"].extend(["_".join(experiment_name.split("_")[2:])] * len(vector))
            df["measure"].extend([suffix] * len(vector))
            df["split"].extend([split] * len(vector))
            df["adv"].extend([adv] * len(vector))
            df["epoch"].extend(np.arange(len(vector)))
            df["wd"].extend([float(experiment_args.weight_decay)] * len(vector))

    df = pd.DataFrame(df)
    df = df >> spread(X.measure, X.score) >> mutate(dec = X.unc + X.rel - X.res)
    df = df >> gather("measure", "score", ["acc", "rel", "res", "score", "unc", "dec", "toplabel_ece"])

    sns.set_style("white", {"font.family": "Times New Roman"})

    grid = sns.relplot(data=df, kind="line", x="epoch", y="score", row="measure", col="experiment",
                       hue="split", style="adv", hue_order=("train", "test"), dashes={"none": "", "pgd": (2, 8)},
                       palette=sns.color_palette("gray", 2), row_order=("score", "res", "rel", "acc", "toplabel_ece"),
                       aspect=1.6, height=1.2, facet_kws={"sharex": True, "sharey": "row", "legend_out": True})

    if experiment_args.dataset == "cifar100":
        ylims = {
            "nll": 5,
            "res": 5,
            "rel": 1,
            "acc": 1.05,
        }
    elif experiment_args.dataset == "cifar":
        ylims = {
            "nll": 2.5,
            "res": 2.0,
            "rel": 0.4,
            "acc": 1.05,
        }

    grid.set_xlabels("Epoch")
#    grid.set_titles("L2 Reg: {col_name}")
    grid.set_titles("{col_name}")
    grid.axes[0][0].set_ylabel("NLL")
    grid.axes[0][0].set_ylim((0, ylims["nll"]))
    grid.axes[1][0].set_ylabel("Resolution")
    grid.axes[1][0].set_ylim((0, ylims["res"]))
    grid.axes[2][0].set_ylabel("Reliability")
    grid.axes[2][0].set_ylim((0, ylims["rel"]))
    grid.axes[3][0].set_ylabel("Accuracy")
    grid.axes[3][0].set_ylim((0, ylims["acc"]))
    grid.tight_layout()
    grid.savefig(f"{args.dir}/loss_curves.pdf")
    plt.show()

