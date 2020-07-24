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
    argparser.add_argument("--score", type=str, default="nll")
    argparser.add_argument("--dir", type=str, default="ckpts")
    args = argparser.parse_args()

    df = defaultdict(list)

    experiment_names = filter(lambda s: os.path.isdir(f"{args.dir}/{s}"), os.listdir(f"{args.dir}"))

    for experiment_name in experiment_names:

        experiment_args = pickle.load(open(f"{args.dir}/{experiment_name}/args.pkl", "rb"))

        for split, suffix in itertools.product(("train", "test"), ("", "acc", "res", "rel", "unc")):

            if suffix == "":
                vector = np.load(f"{args.dir}/{experiment_name}/{split}_{args.score}.npy")
                df["score"].extend(vector)
            elif suffix == "acc":
                vector = np.load(f"{args.dir}/{experiment_name}/{split}_{suffix}.npy")
                df["score"].extend(vector)
            else:
                vector = np.load(f"{args.dir}/{experiment_name}/{split}_{args.score}_{suffix}.npy")
                df["score"].extend(vector)

            suffix = "score" if suffix == "" else suffix
            df["experiment"].extend([experiment_name] * len(vector))
            df["measure"].extend([suffix] * len(vector))
            df["split"].extend([split] * len(vector))
            df["epoch"].extend(np.arange(len(vector)))
            df["wd"].extend([float(experiment_args.weight_decay)] * len(vector))
            if not "num_bins" in experiment_args:
                df["n_bins"].extend([100] * len(vector))
            else:
                df["n_bins"].extend([experiment_args.num_bins] * len(vector))

    df = pd.DataFrame(df)
    df = df >> spread(X.measure, X.score) >> mutate(decomposed = X.unc + X.rel - X.res)
    df = df >> mutate(original = X.score) >> drop(X.score)

    #original = X.score)
#    df = df >> gather("measure", "score", ["acc", "rel", "res", "score", "unc", "dec"])
#    sns.set_style("white", {"font.family": "serif", "font.serif": "Times New Roman"})
#
#    grid = sns.relplot(data=df, kind="line", x="epoch", y="score", row="measure", col="n_bins", hue="split",
#                       palette=sns.color_palette("gray", 2), row_order=("score", "res", "rel", "acc", "dec"),
#                       aspect=1.6, height=1.2, facet_kws={"sharex": True, "sharey": "row", "legend_out": True})
#
#    grid.set_xlabels("Epoch")
#    #grid.set_titles("L2 Reg: {col_name}")
#    grid.set_titles("{col_name}")
#    grid.axes[0][0].set_ylabel("NLL")
#    grid.axes[0][0].set_ylim((0, 4))
#    grid.axes[1][0].set_ylabel("Resolution")
#    grid.axes[1][0].set_ylim((0, 5))
#    grid.axes[2][0].set_ylabel("Reliability")
#    grid.axes[2][0].set_ylim((0, 0.6))
#    grid.axes[3][0].set_ylabel("Accuracy")
#    grid.axes[3][0].set_ylim((0, 1))
#    grid.tight_layout()
#    grid.savefig("./foo.pdf")
#    plt.show()
#
    df = df >> gather("estimator", "score", ["decomposed", "original"])
    #df = df >> gather("measure", "score", ["acc", "rel", "res", "score", "unc"]) >> gather("estimator", "score", ["dec", "score"])
    sns.set_style("white", {"font.family": "Times New Roman"})

    grid = sns.relplot(data=df, kind="line", x="epoch", y="score", row="n_bins", col="split", style="estimator", hue="split",
                       palette=sns.color_palette("gray", 2), #row_order=("score", "res", "rel", "acc"),
                       style_order=("original", "decomposed"),
                       col_order=("train", "test"),
                       hue_order=("train", "test"),
                       aspect=1.6, height=1.2,
                       facet_kws={"sharex": True, "sharey": True, "legend_out": True,
                       "ylim": (0, 5),
                       #"ylim": (0, 2),
                       })

    grid.set_xlabels("Epoch")
    grid.set_ylabels("NLL")
    grid.set_titles("Num Bins: {row_name}")
    grid.tight_layout()
    grid.savefig(f"./{args.dir}/loss_curves.pdf")
    plt.show()

