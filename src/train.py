import logging
import pathlib
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from collections import defaultdict
from torchnet import meter
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.attacks import *
from src.mixup import *
from src.models import *
from src.datasets import get_dataset, get_dim
from src.evaluate import *


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda", type=str)
    argparser.add_argument("--lr", default=0.1, type=float)
    argparser.add_argument("--batch-size", default=64, type=int)
    argparser.add_argument("--num-workers", default=min(os.cpu_count(), 8), type=int)
    argparser.add_argument("--num-epochs", default=120, type=int)
    argparser.add_argument("--print-every", default=20, type=int)
    argparser.add_argument("--save-every", default=50, type=int)
    argparser.add_argument("--experiment-name", default="cifar", type=str)
    argparser.add_argument("--precision", default="half", type=str)
    argparser.add_argument("--model", default="ResNet", type=str)
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--adversary", default=None, type=str)
    argparser.add_argument("--eps", default=8/255, type=float)
    argparser.add_argument("--mixup", action="store_true")
    argparser.add_argument("--weight-decay", default=1e-4, type=float)
    argparser.add_argument("--data-parallel", action="store_true")
    argparser.add_argument("--num-bins", default=100, type=int)
    argparser.add_argument('--output-dir', type=str, default=os.getenv("PT_OUTPUT_DIR"))
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = eval(args.model)(dataset=args.dataset, device=args.device, precision=args.precision)
    model = DataParallelWrapper(model) if args.data_parallel else model

    train_dataset = get_dataset(args.dataset, "train", args.precision)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=False)
    test_dataset = get_dataset(args.dataset, "test", args.precision)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, 
                             num_workers=args.num_workers, pin_memory=False)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=args.weight_decay,
                          nesterov=True)
    annealer = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    time_meter = meter.TimeMeter(unit=False)
    train_loss_meter = meter.AverageValueMeter()

    results = defaultdict(list)

    for epoch in range(args.num_epochs):

        model.train()

        for i, (x, y) in enumerate(train_loader):

            x, y = x.to(args.device), y.to(args.device)
            
            if args.adversary == "fgsm":
                x = fgsm_attack(model, x, y, eps=args.eps, alpha=1.0)
            elif args.adversary == "pgd":
                x = pgd_attack(model, x, y, eps=args.eps, steps=10)
            
            if args.mixup:
                x, y, w = mixup_batch(x, y)
                loss = model.loss(x, y, sample_weights=w).mean()
            else:
                loss = model.loss(x, y).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_meter.add(loss.cpu().data.numpy(), n=1)

            if i % args.print_every == 0:
                logger.info(f"Epoch: {epoch + 1}\t"
                            f"Itr: {i} / {len(train_loader)}\t"
                            f"Loss: {train_loss_meter.value()[0]:.2f}\t"
                            f"Mins: {(time_meter.value() / 60):.2f}\t"
                            f"Experiment: {args.experiment_name}")
                results["train_losses"].append(train_loss_meter.value()[0])
                train_loss_meter.reset()

        if (epoch + 1) % args.save_every == 0:
            save_path = f"{args.output_dir}/{args.experiment_name}/{epoch + 1}/"
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/model_ckpt.torch")

        annealer.step()

        model.eval()
        
        for loader, size, prefix in ((train_loader, len(train_dataset), "train"),
                                     (test_loader, len(test_dataset), "test")):

            top_pred_probs = np.zeros(size)
            top_pred_labels = np.zeros(size)
            overall_preds = np.zeros((size, get_num_labels(args.dataset)))
            overall_logits = np.zeros((size, get_num_labels(args.dataset)))
            overall_labels = np.zeros((size, get_num_labels(args.dataset)))

            for i, (x, y) in enumerate(loader):    
                
                x, y = x.to(args.device), y.to(args.device)

                lower, upper = i * args.batch_size, (i + 1) * args.batch_size
                preds = model.forecast(model.forward(x)).probs.data.cpu().numpy()
                logits = model.forecast(model.forward(x)).logits.data.cpu().numpy()
                top_pred_probs[lower:upper] = np.max(preds, axis=1)
                top_pred_labels[lower:upper] = (np.argmax(preds, axis=1) == y.cpu().numpy()).astype(float)

                overall_preds[lower:upper] = preds
                overall_logits[lower:upper] = logits
                overall_labels[lower:upper] = np.eye(get_num_labels(args.dataset))[y.cpu().data.numpy()]
                
            obs_cdfs, pred_cdfs, bin_cnts = calibration_curve(top_pred_labels, top_pred_probs,
                                                              n_bins=10, raise_on_nan=False)
            results[f"{prefix}_top_calib"].append(calibration_error(obs_cdfs, pred_cdfs, bin_cnts, 2))
            marginal = overall_labels.mean(axis=0, keepdims=True)
            accuracy = (overall_logits.argmax(axis=1) == overall_labels.argmax(axis=1)).mean()

            pred, obs, bin_cnts = discretize_multivar(overall_labels, overall_preds, n_bins=args.num_bins)
            results[f"{prefix}_nll"].append(NLLScore.score(overall_logits, overall_labels, logits=True).mean())
            results[f"{prefix}_nll_rel"].append(NLLScore.reliability(obs, pred, bin_cnts))
            results[f"{prefix}_nll_res"].append(NLLScore.resolution(obs, marginal, bin_cnts))
            results[f"{prefix}_nll_unc"].append(NLLScore.uncertainty(marginal, np.sum(bin_cnts)))
            results[f"{prefix}_brier"].append(BrierScore.score(overall_preds, overall_labels).mean())
            results[f"{prefix}_brier_rel"].append(BrierScore.reliability(obs, pred, bin_cnts))
            results[f"{prefix}_brier_res"].append(BrierScore.resolution(obs, marginal, bin_cnts))
            results[f"{prefix}_brier_unc"].append(BrierScore.uncertainty(marginal, np.sum(bin_cnts)))
            results[f"{prefix}_acc"].append(accuracy)

    pathlib.Path(f"{args.output_dir}/{args.experiment_name}").mkdir(parents=True, exist_ok=True)
    save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    torch.save(model.state_dict(), save_path)
    args_path = f"{args.output_dir}/{args.experiment_name}/args.pkl"
    pickle.dump(args, open(args_path, "wb"))

    for k, v in results.items():
        save_path = f"{args.output_dir}/{args.experiment_name}/{k}.npy"
        np.save(save_path, np.array(v))

