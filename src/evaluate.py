import numpy as np
import scipy as sp
import scipy.special
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from src.calib import *
from src.datasets import *
from src.scores import *


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dir", type=str, default="ckpts")
    argparser.add_argument("--experiment-name", type=str, default="cifar")
    argparser.add_argument("--num-images-saved", type=int, default=1000)
    argparser.add_argument("--dataset", type=str, default=None)
    argparser.add_argument("--save-examples", action="store_true")
    args = argparser.parse_args()

    folder = Path(f"{args.dir}/{args.experiment_name}/")
    folder.mkdir(parents=True, exist_ok=True)

    #train_losses = np.load(f"ckpts/{args.experiment_name}/train_losses.npy")
    #test_losses = np.load(f"ckpts/{args.experiment_name}/test_losses.npy")

#    delta = len(train_losses) // len(test_losses)
#    plt.figure(figsize=(6, 3))
#    plt.plot(train_losses, color="black", label="Train")
#    plt.plot(np.arange(0, len(train_losses), delta), test_losses, color="grey", label="Test")
#    plt.xlabel("Training Iterations")
#    plt.ylabel("Loss")
#    plt.legend()
#    plt.tight_layout()
#    plt.savefig(f"{folder}/loss_curve.png")

    preds = np.load(f"{args.dir}/{args.experiment_name}/preds.npy")
    preds_adv = np.load(f"{args.dir}/{args.experiment_name}/preds_adv.npy")
    labels = np.load(f"{args.dir}/{args.experiment_name}/labels.npy")
    cams = np.load(f"{args.dir}/{args.experiment_name}/cams.npy")
    dataset = args.dataset if args.dataset else args.experiment_name.split("_")[0]
    test_dataset = get_dataset(dataset, "test", precision="float")
    label_names = get_label_names(dataset)
    labels_one_hot = np.eye(get_num_labels(dataset))[labels]

    print(f"== Accuracy: {np.mean(np.argmax(preds, axis=1) == labels):.3f}")
    print(f"== Adv Accuracy: {np.mean(np.argmax(preds_adv, axis=1) == labels):.3f}")

    print(f"== Macro-averaged AUPRC: {average_precision_score(labels_one_hot, preds, 'macro'):.3f}")
    print(f"== Micro-averaged AUPRC: {average_precision_score(labels_one_hot, preds, 'micro'):.3f}")
    print("== Per-class AUPRCs:")
    for label_id, label_name in enumerate(label_names):
        print(f"  {label_name}: {average_precision_score(labels == label_id, preds[:, label_id]):.3f}")

    print("== Top label calibration error (over top prediction)")
    print("  (Normal)")
    top_pred_cats = np.argmax(preds, axis=1)
    rng = np.arange(len(preds))
    obs_cdfs, pred_cdfs, bin_cnts = calibration_curve(labels_one_hot[rng, top_pred_cats],
                                                      preds[rng, top_pred_cats], n_bins=10,
                                                      raise_on_nan=False)
    print(f"  L2 Calibration Error: {calibration_error(obs_cdfs, pred_cdfs, bin_cnts):.2f}")
    print(f"  L1 Calibration Error: {calibration_error(obs_cdfs, pred_cdfs, bin_cnts, 1):.2f}")

    print("  (Adv)")
    top_pred_cats = np.argmax(preds_adv, axis=1)
    rng = np.arange(len(preds))
    obs_cdfs, pred_cdfs, bin_cnts = calibration_curve(labels_one_hot[rng, top_pred_cats],
                                                      preds_adv[rng, top_pred_cats], n_bins=10,
                                                      raise_on_nan=False)
    print(f"  L2 Calibration Error: {calibration_error(obs_cdfs, pred_cdfs, bin_cnts):.2f}")
    print(f"  L1 Calibration Error: {calibration_error(obs_cdfs, pred_cdfs, bin_cnts, 1):.2f}")

    print("== Marginal calibration error (over all classes)")
    print("  (Normal)")
    obs_cdfs, pred_cdfs, bin_cnts = calibration_curve(labels_one_hot.ravel(), preds.ravel())
    print(f"  L2 Calibration Error: {calibration_error(obs_cdfs, pred_cdfs, bin_cnts):.2f}")
    print(f"  L1 Calibration Error: {calibration_error(obs_cdfs, pred_cdfs, bin_cnts, 1):.2f}")
    print("  (Adv)")
    obs_cdfs, pred_cdfs, bin_cnts = calibration_curve(labels_one_hot.ravel(), preds_adv.ravel(), raise_on_nan=False)
    print(f"  L2 Calibration Error: {calibration_error(obs_cdfs, pred_cdfs, bin_cnts):.2f}")
    print(f"  L1 Calibration Error: {calibration_error(obs_cdfs, pred_cdfs, bin_cnts, 1):.2f}")

    plt.figure(figsize=(5, 5))
    plt.scatter(pred_cdfs, obs_cdfs, color="black")
    plt.plot((0, 1), (0, 1), "--", color="grey")
    plt.xlim((0, 1))
    plt.xlabel("Expected CDF")
    plt.ylim((0, 1))
    plt.ylabel("Observed CDF")
    plt.savefig(f"{folder}/calib.png")

    if not args.save_examples:
        exit()

    print("== Saving examples...")
    plt.figure(figsize=(8, 4))
    for i in tqdm(range(args.num_images_saved)):
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(cams[i])
        plt.title(f"Pred: {label_names[preds[i].argmax()]} {preds[i].max():.2f}")
        plt.subplot(1, 2, 2)
        plt.imshow(test_dataset[i][0].numpy().transpose(1, 2, 0))
        plt.title(f"Truth: {label_names[int(labels[i])]}")
        folder = Path(f"out/{args.experiment_name}/{label_names[int(labels[i])]}/")
        folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{folder}/{i}.png")

