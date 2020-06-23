import numpy as np
from argparse import ArgumentParser
from easydict import EasyDict
from matplotlib import pyplot as plt


if __name__ == "__main__":
    
    argparser = ArgumentParser()
    argparser.add_argument("--experiment-name", type=str, default="cifar100")
    argparser.add_argument("--score", type=str, default="nll")
    args = argparser.parse_args()

    results = EasyDict()
    for prefix in ("train", "test"):
        for k in (f"{prefix}_{args.score}", f"{prefix}_{args.score}_rel", f"{prefix}_acc",
                  f"{prefix}_{args.score}_res", f"{prefix}_{args.score}_unc"):
            results[k.replace(f"_{args.score}", "")] = np.load(f"ckpts/{args.experiment_name}/{k}.npy")

    decomp = results.test_unc - results.test_res + results.test_rel

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(results.train, label="Train", color="grey")
    plt.plot(results.test, label="Test", color="black")
    plt.plot(results.train_unc - results.train_res + results.train_rel, label="Train", color="grey", linestyle="--")
    plt.plot(results.test_unc - results.test_res + results.test_rel, label="Test", color="black", linestyle="--")
    plt.legend()
    plt.title("SCORE")
    plt.ylim((0., 4.))
    plt.subplot(2, 2, 2)
    plt.plot(results.train_rel, label="Train", color="grey")
    plt.plot(results.test_rel, label="Test", color="black")
    plt.legend()
    plt.title("REL")
    plt.ylim((0., 1.))
    plt.subplot(2, 2, 3)
    plt.plot(results.train_res, label="Train", color="grey")
    plt.plot(results.test_res, label="Test", color="black")
    plt.legend()
    plt.title("RES")
    plt.ylim((0., 5.))
    plt.subplot(2, 2, 4)
    plt.legend()
    plt.plot(results.train_acc, label="Train", color="grey")
    plt.plot(results.test_acc, label="Test", color="black")
    plt.legend()
    plt.title("ACC")
    plt.ylim((0., 1.))
    plt.tight_layout()
    plt.show()

