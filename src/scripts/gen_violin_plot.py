import numpy as np
import pandas as pd
import scipy as sp
import scipy.special
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
    argparser.add_argument("--adv", action="store_true")
    argparser.add_argument("--eval", action="store_true")
    argparser.add_argument("--inspect", action="store_true")
    args = argparser.parse_args()

    df = defaultdict(list)
    experiment_names = filter(lambda s: os.path.isdir(f"{args.dir}/{s}"), os.listdir(f"{args.dir}"))

    for experiment_name in experiment_names:

        temperatures = np.load(f"{args.dir}/{experiment_name}/temperatures.npy")
        temperatures_labels = np.load(f"{args.dir}/{experiment_name}/temperatures_labels.npy")

        q = np.load(f"{args.dir}/{experiment_name}/q.npy")
        logits_clean = np.load(f"{args.dir}/{experiment_name}/logits.npy")
        logits_pgd = np.load(f"{args.dir}/{experiment_name}/logits_pgd.npy")
        logits_max_conf = np.load(f"{args.dir}/{experiment_name}/logits_max_conf.npy")

        adv_labels = ("clean", "pgd", "max_conf")

        for (T_label, T), (adv_label, logits) in itertools.product(zip(temperatures_labels, temperatures),
                                                                   zip(adv_labels, (logits_clean, logits_pgd, logits_max_conf))):

            p = sp.special.softmax(logits / T, axis=1)

            mask_correct = p.argmax(axis=1) == q.argmax(axis=1)
            p_correct = p.max(axis=1)[mask_correct]
            p_incorrect = p.max(axis=1)[~mask_correct]
            assert p_correct.min() >= 0.1

            #p_correct = p[np.arange(len(p)), q.argmax(axis=1)]
            #p_incorrect = p.max(axis=1)

            df["experiment_name"].extend(["_".join(experiment_name.split("_")[2:])] * len(p_correct))
            df["temperature"].extend([T_label] * len(p_correct))
            df["adversary"].extend([adv_label] * len(p_correct))
            df["p_top_label"].extend(p_correct)
            df["correct"].extend(["yes"] * len(p_correct))

            df["experiment_name"].extend(["_".join(experiment_name.split("_")[2:])] * len(p_incorrect))
            df["temperature"].extend([T_label] * len(p_incorrect))
            df["adversary"].extend([adv_label] * len(p_incorrect))
            df["p_top_label"].extend(p_incorrect)
            df["correct"].extend(["no"] * len(p_incorrect))

    df = pd.DataFrame(df)
    sns.set_style("white", {"font.family": "Times New Roman"})

    grid = sns.catplot(data=df, kind="violin", x="adversary", y="p_top_label", hue="correct",
                       row="temperature", row_order=("vanilla", "adv", "max_conf"),
                       col="experiment_name", col_order=sorted(set(df["experiment_name"])),
                       split=True, height=2, aspect=1.6, inner=None, scale="count",
                       palette=sns.color_palette("gray", 2), legend=True,
                       facet_kws={"sharex": True, "sharey": "row", "legend_out": True})

    grid = grid.add_legend()
    grid.set_xlabels("Adversary")
    grid.set_titles("{col_name}")
    grid.axes[0][0].set_ylabel("T = Clean")
    grid.axes[1][0].set_ylabel("T = PGD")
    grid.axes[2][0].set_ylabel("T = MaxConf")
    breakpoint()
    plt.show()
#
#    grid = sns.relplot(data=df, kind="line", x="epoch", y="value", row="key", col="experiment",
#                       hue="split", hue_order=("train", "test"),
#                       style="adv", dashes={"none": "", "pgd": (2, 4)},
#                       size="interval", sizes=(1, 1),
#                       palette=sns.color_palette("gray", 2),
#                       row_order=("nll", "acc", "toplabel_ece", "consistency_toplabel_ece"),
#                       col_order=sorted(set(df["experiment"])),
#                       aspect=1.6, height=1.2, facet_kws={"sharex": True, "sharey": "row", "legend_out": True})
#
