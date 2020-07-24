import numpy as np
import pandas as pd
import itertools
import pickle
import seaborn as sns
import matplotlib as mpl
from argparse import ArgumentParser
from collections import defaultdict
from matplotlib import pyplot as plt
from dfply import *


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dir", type=str, default="cifar100")
    argparser.add_argument("--score", type=str, default="nll")
    args = argparser.parse_args()

    df = defaultdict(list)

    EXPERIMENTS = ("cifar100_wd_1e-5",)

    for experiment_name in EXPERIMENTS:

        experiment_args = pickle.load(open(f"{args.dir}/{experiment_name}/args.pkl", "rb"))

        for split, adv, suffix in itertools.product(("train", "test"), ("",),
                                                    ("", "acc", "res", "rel", "unc")):

            if suffix == "": # plug-in estimator
                vector = np.load(f"{args.dir}/{experiment_name}/{split}{adv}_{args.score}.npy")
                df["score"].extend(vector)
            elif suffix == "acc": # accuracy
                vector = np.load(f"{args.dir}/{experiment_name}/{split}{adv}_{suffix}.npy")
                df["score"].extend(vector)
            else:
                vector = np.load(f"{args.dir}/{experiment_name}/{split}{adv}_{args.score}_{suffix}.npy")
                df["score"].extend(vector)

            suffix = "score" if suffix == "" else suffix
            adv = "none" if adv == "" else "pgd"

            df["experiment"].extend([experiment_name[13:]] * len(vector))
            df["measure"].extend([suffix] * len(vector))
            df["split"].extend([split] * len(vector))
            df["epoch"].extend(np.arange(len(vector)))
            df["wd"].extend([float(experiment_args.weight_decay)] * len(vector))

    df = pd.DataFrame(df)
    df = df >> spread(X.measure, X.score) >> mutate(dec = X.unc + X.rel - X.res)
    df = df >> gather("measure", "score", ["acc", "rel", "res", "score", "unc", "dec"])

    sns.set_style("white", {"font.family": "Times New Roman"})

    grid = sns.relplot(data=df, kind="line", x="epoch", y="score", row="experiment", col="measure",
                       hue="split", hue_order=("train", "test"), dashes={"none": "", "pgd": (2, 5)},
                       palette=sns.color_palette("gray", 2), col_order=("score", "res", "rel", "acc"),
                       aspect=1.6, height=1.4, facet_kws={"sharex": True, "sharey": False, "legend_out": True})

    grid.set_xlabels("Epoch")
    grid.set_titles("{row_name}")
    grid.axes[0][0].set_title("NLL")
    grid.axes[0][0].set_ylim((0, 5))
    grid.axes[0][0].set_ylabel("")
    grid.axes[0][1].set_title("Resolution")
    grid.axes[0][1].set_ylim((0, 5))
    grid.axes[0][2].set_title("Reliability")
    grid.axes[0][2].set_ylim((0, 1))
    grid.axes[0][3].set_title("Accuracy")
    grid.axes[0][3].set_ylim((0, 1.05))
    grid.tight_layout()
    grid.savefig("./figs/figure_1a.pdf")
    plt.show()

