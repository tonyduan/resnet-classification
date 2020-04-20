import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from src.datasets import *


if __name__ == "__main__": 

    argparser = ArgumentParser()
    argparser.add_argument("--experiment-name", type=str, default="cifar")
    argparser.add_argument("--dataset", type=str, default="cifar")
    argparser.add_argument("--num-images-saved", type=int, default=1000)
    args = argparser.parse_args()

    preds = np.load(f"ckpts/{args.experiment_name}/preds.npy")
    labels = np.load(f"ckpts/{args.experiment_name}/labels.npy")
    cams = np.load(f"ckpts/{args.experiment_name}/cams.npy")
    test_dataset = get_dataset(args.dataset, "test", precision="float")
    label_names = get_label_names("cifar")
    labels_one_hot = np.eye(get_num_labels(args.dataset))[labels]

    print("== Accuracy:", np.mean(np.argmax(preds, axis=1) == labels))
    print("== Macro-averaged AUPRC:", average_precision_score(labels_one_hot, preds, "macro"))
    print("== Micro-averaged AUPRC:", average_precision_score(labels_one_hot, preds, "micro"))
    print("== Per-class AUPRCs:")
    for label_id, label_name in enumerate(label_names):
        print(f"  {label_name}:", average_precision_score(labels == label_id, preds[:, label_id]))
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
        folder = Path(f"out/{args.dataset}/{label_names[int(labels[i])]}/")
        folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{folder}/{i}.png")

