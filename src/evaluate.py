import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from src.scores import discretize_multivar, NLLScore, BrierScore
from src.calib import calibration_curve, calibration_error
from src.datasets import *
from src.utils import bootstrap_resample


def evaluate_snapshot(q, p, p_logits):
    """
    q: one-hot labels
    p: predictions
    p_logits: logits for predictions
    """
    # marginal needed for discretization
    #pi_tilde = q.mean(axis=0, keepdims=True)

    # top-label calibration
    top_pred_labels = (p_logits.argmax(axis=1) == q.argmax(axis=1)).astype(float)
    top_pred_probs = p.max(axis=1)
    obs_cdfs, pred_cdfs, bin_cnts = calibration_curve(top_pred_labels, top_pred_probs,
                                                      n_bins=15, raise_on_nan=False)
    toplabel_ece = calibration_error(obs_cdfs, pred_cdfs, bin_cnts, p=1)

    # marginal calibration
    obs_cdfs, pred_cdfs, bin_cnts = calibration_curve(q.ravel(), p.ravel(),
                                                      n_bins=15, raise_on_nan=False)
    marginal_ece = calibration_error(obs_cdfs, pred_cdfs, bin_cnts, p=1)

    # scoring rules
    #pi, gamma, bin_cnts = discretize_multivar(q, p)

    # consistency resampling
    p_cumsums, rngs = np.cumsum(p, axis=1), np.random.rand(len(p))
    p_cumsums[:, -1] = 1.  # fix numerical instability
    q_consistency = [np.searchsorted(p_cumsum, rng) for p_cumsum, rng in zip(p_cumsums, rngs)]
    q_consistency = np.eye(q.shape[1])[np.array(q_consistency)]

    # consistency top-label calibration
    top_pred_labels = (p_logits.argmax(axis=1) == q_consistency.argmax(axis=1)).astype(float)
    top_pred_probs = p.max(axis=1)
    obs_cdfs, pred_cdfs, bin_cnts = calibration_curve(top_pred_labels, top_pred_probs,
                                                      n_bins=15, raise_on_nan=False)
    consistency_toplabel_ece = calibration_error(obs_cdfs, pred_cdfs, bin_cnts, p=1)

    # marginal calibration
    obs_cdfs, pred_cdfs, bin_cnts = calibration_curve(q_consistency.ravel(), p.ravel(),
                                                      n_bins=15, raise_on_nan=False)
    consistency_marginal_ece = calibration_error(obs_cdfs, pred_cdfs, bin_cnts, p=1)

    return {
        "acc": (p_logits.argmax(axis=1) == q.argmax(axis=1)).mean(),
        "nll": NLLScore.score(p_logits, q, logits=True).mean(),
        #"nll_res": NLLScore.reliability(pi, gamma, bin_cnts),
        #"nll_rel": NLLScore.resolution(pi, pi_tilde, bin_cnts),
        #"nll_unc": NLLScore.uncertainty(pi_tilde, np.sum(bin_cnts)),
        "brier": BrierScore.score(p, q).mean(),
        #"brier_rel": BrierScore.reliability(pi, gamma, bin_cnts),
        #"brier_res": BrierScore.resolution(pi, pi_tilde, bin_cnts),
        #"brier_unc": BrierScore.uncertainty(pi_tilde, np.sum(bin_cnts)),
        "toplabel_ece": toplabel_ece,
        "marginal_ece": marginal_ece,
        "consistency_toplabel_ece": consistency_toplabel_ece,
        "consistency_marginal_ece": consistency_marginal_ece,
    }


def bootstrap_snapshot(q, p, p_logits, num_samples=1000):

    bootstrap_df = defaultdict(list)

    for _ in tqdm(range(num_samples)):
        idxs = bootstrap_resample(q.argmax(axis=1))
        for k, v in evaluate_snapshot(q[idxs], p[idxs], p_logits[idxs]).items():
            bootstrap_df[k].append(v)

    bootstrap_df = pd.DataFrame(bootstrap_df)
    lower = {f"lower_{k}": v for k, v in bootstrap_df.quantile(q=0.025).to_dict().items()}
    upper = {f"upper_{k}": v for k, v in bootstrap_df.quantile(q=0.975).to_dict().items()}
    return {**lower, **upper}  # merges the two dictionaries


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--output-dir", type=str, default="ckpts")
    argparser.add_argument("--experiment-name", type=str, default="cifar")
    argparser.add_argument("--num-images-saved", type=int, default=1000)
    argparser.add_argument("--dataset", type=str, default=None)
    argparser.add_argument("--save-examples", action="store_true")
    args = argparser.parse_args()

    # load snapshots
    logits = np.load(f"{args.output_dir}/{args.experiment_name}/logits.npy")
    logits_adv = np.load(f"{args.output_dir}/{args.experiment_name}/logits_adv.npy")
    p = np.load(f"{args.output_dir}/{args.experiment_name}/p.npy")
    p_adv = np.load(f"{args.output_dir}/{args.experiment_name}/p_adv.npy")
    q = np.load(f"{args.output_dir}/{args.experiment_name}/q.npy")
    cams = np.load(f"{args.output_dir}/{args.experiment_name}/cams.npy")
    q_flat = q.argmax(axis=1)

    # load label names for interpretability
    dataset = args.dataset if args.dataset else args.experiment_name.split("_")[0]
    label_names = get_label_names(dataset)

    print(f"== Macro-averaged AUPRC: {average_precision_score(q, p, 'macro'):.3f}")
    print(f"== Micro-averaged AUPRC: {average_precision_score(q, p, 'micro'):.3f}")
    print("== Per-class AUPRCs:")
    for label_id, label_name in enumerate(label_names):
        print(f"  {label_name}: {average_precision_score(q_flat == label_id, p[:, label_id]):.3f}")

    # save post-hoc final report
    eval_report = evaluate_snapshot(q, p, logits)
    adv_eval_report = evaluate_snapshot(q, p_adv, logits_adv)
    bootstrap_report = bootstrap_snapshot(q, p, logits)
    adv_bootstrap_report = bootstrap_snapshot(q, p_adv, logits_adv)

    df = defaultdict(list)

    for k, v in itertools.chain(eval_report.items(), adv_eval_report.items(),
                                bootstrap_report.items(), adv_bootstrap_report.items()):
        df[k].append(v)
        print(f"== {k}: {v:.3f}")

    df = pd.DataFrame(df)
    df.to_csv(f"{args.output_dir}/{args.experiment_name}/eval_results.csv")

    if not args.save_examples:
        exit()

    test_dataset = get_dataset(dataset, "test", precision="float")
    print("== Saving examples...")
    plt.figure(figsize=(8, 4))
    for i in tqdm(range(args.num_images_saved)):
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(cams[i])
        plt.title(f"Pred: {label_names[p[i].argmax()]} {p[i].max():.3f}")
        plt.subplot(1, 2, 2)
        plt.imshow(test_dataset[i][0].numpy().transpose(1, 2, 0))
        plt.title(f"Truth: {label_names[int(q_flat[i])]}")
        folder = Path(f"out/{args.experiment_name}/{label_names[int(q_flat[i])]}/")
        folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{folder}/{i}.png")
