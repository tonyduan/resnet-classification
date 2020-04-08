import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from src.datasets import *


if __name__ == "__main__": 

    argparser = ArgumentParser()
    argparser.add_argument("--experiment-name", type=str, default="cifar")
    argparser.add_argument("--dataset", type=str, default="cifar")
    args = argparser.parse_args()

    preds = np.load(f"ckpts/{args.experiment_name}/preds.npy")
    labels = np.load(f"ckpts/{args.experiment_name}/labels.npy")
    cams = np.load(f"ckpts/{args.experiment_name}/cams.npy")
    test_dataset = get_dataset(args.dataset, "test", precision="float")

    print("Accuracy:", np.mean(np.argmax(preds, axis=1) == labels))

    idx = 20

    for i in range(4):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(cams[idx + i])
        plt.subplot(1, 2, 2)
        plt.imshow(test_dataset[idx + i][0].numpy().transpose(1, 2, 0))

    plt.show()
