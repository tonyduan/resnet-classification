import logging
import pathlib
import pickle
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from argparse import ArgumentParser
from collections import defaultdict
from torchnet import meter
from torch.distributions import Categorical, kl_divergence
from torch.utils.data import Subset
from src.attacks import fgsm_attack, pgd_attack
from src.mixup import mixup_batch
from src.models import *
from src.utils import split_hold_out_set
from src.datasets import get_dataset, get_dataloader, get_num_labels
from src.evaluate import evaluate_snapshot, bootstrap_snapshot


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
    argparser.add_argument("--precision", default="float", type=str)
    argparser.add_argument("--model", default="ResNet", type=str)
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--eval-dataset", default=None, type=str)
    argparser.add_argument("--adversary", default=None, type=str)
    argparser.add_argument("--eps", default=8 / 255, type=float)
    argparser.add_argument("--mixup", default=False, type=bool)
    argparser.add_argument("--ccat", action="store_true")
    argparser.add_argument("--weight-decay", default=1e-4, type=float)
    argparser.add_argument("--data-parallel", action="store_true")
    argparser.add_argument("--num-bins", default=None, type=int)
    argparser.add_argument("--use-val-set", action="store_true")
    argparser.add_argument('--output-dir', type=str, default=os.getenv("PT_OUTPUT_DIR"))
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = eval(args.model)(dataset=args.dataset, device=args.device, precision=args.precision)
    model = DataParallelWrapper(model) if args.data_parallel else model

    if not args.use_val_set:

        train_dataset = get_dataset(args.dataset, "train", args.precision)
        train_loader = get_dataloader(train_dataset, "train", args.batch_size, args.num_workers)

        _, subset_idxs = split_hold_out_set(train_dataset.targets, 10000)
        train_subset_dataset = Subset(train_dataset, list(subset_idxs))
        train_subset_loader = get_dataloader(train_subset_dataset, "train",
                                             args.batch_size, args.num_workers)

        test_dataset = get_dataset(args.eval_dataset or args.dataset, "test", args.precision)
        test_loader = get_dataloader(test_dataset, "test", args.batch_size, args.num_workers)

        eval_loaders_and_datasets = ((train_subset_loader, len(train_subset_dataset), "train"),
                                     (test_loader, len(test_dataset), "test"))

    else:

        train_dataset = get_dataset(args.dataset, "train_train", args.precision)
        train_loader = get_dataloader(train_dataset, "train", args.batch_size, args.num_workers)

        targets = np.array(train_dataset.dataset.targets)[train_dataset.indices]
        _, subset_idxs = split_hold_out_set(targets, 10000)
        train_subset_dataset = Subset(train_dataset, list(subset_idxs))
        train_subset_loader = get_dataloader(train_subset_dataset, "train",
                                             args.batch_size, args.num_workers)

        val_dataset = get_dataset(args.dataset, "train_val", args.precision)
        val_loader = get_dataloader(val_dataset, "val", args.batch_size, args.num_workers)

        test_dataset = get_dataset(args.eval_dataset or args.dataset, "test", args.precision)
        test_loader = get_dataloader(test_dataset, "test", args.batch_size, args.num_workers)

        eval_loaders_and_datasets = ((train_subset_loader, len(train_subset_dataset), "train"),
                                     (val_loader, len(val_dataset), "val"),
                                     (test_loader, len(test_dataset), "test"))

    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=args.weight_decay,
                          nesterov=True)
    annealer = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    time_meter = meter.TimeMeter(unit=False)
    loss_meter = meter.AverageValueMeter()

    loss_curve = []
    results = defaultdict(list)

    # default to 2x the number of output categories
    if args.num_bins is None:
        args.num_bins = 2 * get_num_labels(args.dataset)

    if args.ccat:
        uniform_categorical = Categorical(probs=torch.ones(get_num_labels(args.dataset),
                                          device=args.device))

    for epoch in range(args.num_epochs):

        model.train()

        for i, (x, y) in enumerate(train_loader):

            x, y = x.to(args.device), y.to(args.device)
            x_orig = x

            if args.adversary == "fgsm":
                x = fgsm_attack(model, x, y, eps=args.eps, alpha=1.0)
            elif args.adversary == "pgd":
                x = pgd_attack(model, x, y, eps=args.eps, steps=10)

            if args.mixup:
                x, y, w = mixup_batch(x, y)
                loss = model.loss(x, y, sample_weights=w).mean()
            elif args.ccat:
                batch_cutoff = len(x) // 2
                eps = (x_orig[batch_cutoff:] - x[batch_cutoff:]).norm(p=np.inf, dim=(1, 2, 3))
                lambd = (1 - torch.min(eps / args.eps, torch.ones_like(eps))) ** 10
                forecast_adv = model.forecast(model.forward(x[batch_cutoff:]))
                loss_clean = model.loss(x_orig[:batch_cutoff], y[:batch_cutoff])
                loss_adv = (lambd * model.loss(x[batch_cutoff:], y[batch_cutoff:])
                            + (1 - lambd) * kl_divergence(uniform_categorical, forecast_adv))
                loss = 0.5 * loss_clean.mean() + 0.5 * loss_adv.mean()
            else:
                loss = model.loss(x, y).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.cpu().data.numpy(), n=1)

            if i % args.print_every == 0:
                logger.info(f"Epoch: {epoch + 1}\t"
                            f"Itr: {i} / {len(train_loader)}\t"
                            f"Loss: {loss_meter.value()[0]:.2f}\t"
                            f"Mins: {(time_meter.value() / 60):.2f}\t"
                            f"Experiment: {args.experiment_name}")
                loss_curve.append(loss_meter.value()[0])
                loss_meter.reset()

        if (epoch + 1) % args.save_every == 0:
            save_path = f"{args.output_dir}/{args.experiment_name}/{epoch + 1}/"
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/model_ckpt.torch")

        annealer.step()
        model.eval()

        for loader, size, prefix in eval_loaders_and_datasets:

            p = np.zeros((size, get_num_labels(args.dataset)))
            p_adv = np.zeros((size, get_num_labels(args.dataset)))
            p_logits = np.zeros((size, get_num_labels(args.dataset)))
            p_logits_adv = np.zeros((size, get_num_labels(args.dataset)))
            q = np.zeros((size, get_num_labels(args.dataset)))

            for i, (x, y) in enumerate(loader):

                x, y = x.to(args.device), y.to(args.device)

                lower, upper = i * args.batch_size, (i + 1) * args.batch_size
                q[lower:upper] = np.eye(get_num_labels(args.dataset))[y.cpu().data.numpy()]
                p[lower:upper] = model.forecast(model.forward(x)).probs.data.cpu().numpy()
                p_logits[lower:upper] = model.forecast(model.forward(x)).logits.data.cpu().numpy()

                if args.adversary is not None:

                    x_adv = pgd_attack(model, x, y, eps=args.eps, steps=20)
                    p_adv[lower:upper] = model.forecast(model.forward(x_adv)).probs.data.cpu().numpy()
                    p_logits_adv[lower:upper] = model.forecast(model.forward(x_adv)).logits.data.cpu().numpy()

            for k, v in evaluate_snapshot(q, p, p_logits).items():
                results[f"{prefix}_{k}"].append(v)

            for k, v in bootstrap_snapshot(q, p, p_logits).items():
                results[f"{prefix}_{k}"].append(v)

            if args.adversary is not None:

                for k, v in evaluate_snapshot(q, p_adv, p_logits_adv).items():
                    results[f"{prefix}_adv_{k}"].append(v)

                for k, v in bootstrap_snapshot(q, p_adv, p_logits_adv).items():
                    results[f"{prefix}_adv_{k}"].append(v)

    pathlib.Path(f"{args.output_dir}/{args.experiment_name}").mkdir(parents=True, exist_ok=True)

    np.save(f"{args.output_dir}/loss_curve.npy", np.array(loss_curve))
    torch.save(model.state_dict(), f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch")

    with open(f"{args.output_dir}/{args.experiment_name}/args.pkl", "wb") as args_file:
        pickle.dump(args, args_file)

    df = pd.DataFrame(results)
    df.to_csv(f"{args.output_dir}/{args.experiment_name}/results.csv", index=False)
