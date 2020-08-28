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

    EXPERIMENTS = ("cifar100_pgd_wd_1e-5_mixup_0.0",)
    eval_df = pd.read_csv(f"{args.dir}/{EXPERIMENTS[0]}/eval_results.csv")
    eval_df = eval_df >> mask(X.temperature_label == "vanilla")

    for experiment_name in EXPERIMENTS:

        experiment_args = pickle.load(open(f"{args.dir}/{experiment_name}/args.pkl", "rb"))
        experiment_df = pd.read_csv(f"{args.dir}/{experiment_name}/results.csv")

        for split, adv, interval, key in itertools.product(("train", "test"), ("", "_adv"),
                                                           ("lower", "upper"),
                                                           ("nll", "acc", "toplabel_ece", "consistency_toplabel_ece")):

            if interval == "":
                col = f"{split}{adv}_{key}"
            else:
                col = f"{split}{adv}_{interval}_{key}"

            adv = "none" if adv == "" else "pgd"
            df["adv"].extend([adv] * len(experiment_df))
            df["experiment"].extend(["_".join(experiment_name.split("_")[1:])] * len(experiment_df))
            df["interval"].extend([interval] * len(experiment_df))
            df["split"].extend([split] * len(experiment_df))
            df["key"].extend([key] * len(experiment_df))
            df["value"].extend(experiment_df[col])
            df["epoch"].extend(np.arange(len(experiment_df)) + 1)

    df = pd.DataFrame(df)

    print(len(df))
    df >>= mask(~((X.key == "toplabel_ece") & (X.interval == "lower")))
    df >>= mask(~((X.key == "consistency_toplabel_ece") & (X.interval == "lower")))
    df >>= mask(~((X.key == "accuracy") & (X.interval == "upper")))
    df >>= mask(~((X.key == "nll") & (X.interval == "lower")))
    print(len(df))

    sns.set_style("white", {"font.family": "Times New Roman"})
    grid = sns.relplot(data=df, kind="line", x="epoch", y="value", row="experiment", col="key",
                       hue="split", hue_order=("train", "test"),
                       style="adv",
                       dashes={"none": "", "pgd": (2, 5),},
                       palette=sns.color_palette("gray", 2),
                       col_order=("nll", "acc", "toplabel_ece", "consistency_toplabel_ece"),
                       size=1,
                       aspect=1.2, height=1.6,
                       facet_kws={"sharex": True, "sharey": False, "legend_out": True})

    for item in grid._legend.texts[3:4]:
        item.set_visible(0)
    for item in grid._legend.legendHandles[3:4]:
        item.set_visible(0)

    grid.set_xlabels("Epoch")
    grid.set_titles("{row_name}")
    grid.axes[0][0].set_title("NLL")
    grid.axes[0][0].set_ylim((0, 5))
    grid.axes[0][0].set_ylabel("")
    grid.axes[0][1].set_title("Accuracy")
    grid.axes[0][1].set_ylim((0, 1.05))
    grid.axes[0][2].set_title("ECE")
    grid.axes[0][2].set_ylim((0, 0.4))
    grid.axes[0][3].set_title("Consistency ECE")
    grid.axes[0][3].set_ylim((0, 0.02))
    grid.tight_layout()
    grid.savefig("./figs/loss_curve_adv.pdf")
    plt.show()

