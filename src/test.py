import numpy as np
import pathlib
import os
import torch
from argparse import ArgumentParser
from torch.utils.data import Subset
from tqdm import tqdm
from src.models import *
from src.datasets import get_num_labels, get_dataset, get_dataloader
from src.attacks import pgd_attack
from src.calib import temperature_scale


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda", type=str)
    argparser.add_argument("--batch-size", default=64, type=int)
    argparser.add_argument("--num-workers", default=min(os.cpu_count(), 8), type=int)
    argparser.add_argument("--dataset-skip", default=1, type=int)
    argparser.add_argument("--eps", default=8 / 255, type=float)
    argparser.add_argument("--experiment-name", default="cifar", type=str)
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--eval-dataset", default=None, type=str)
    argparser.add_argument("--data-parallel", action="store_true")
    argparser.add_argument("--model", default="ResNet", type=str)
    argparser.add_argument("--precision", default="float", type=str)
    argparser.add_argument("--temperature-scale", action="store_true")
    argparser.add_argument("--adv-temperature-scale", action="store_true")
    argparser.add_argument("--output-dir", type=str, default=os.getenv("PT_OUTPUT_DIR"))
    argparser.add_argument("--save-path", type=str, default=None)
    args = argparser.parse_args()

    if not args.save_path:
        save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    else:
        save_path = args.save_path

    model = eval(args.model)(dataset=args.dataset, device=args.device, precision=args.precision)
    model = DataParallelWrapper(model) if args.data_parallel else model
    saved_dict = torch.load(save_path)
    model.load_state_dict(saved_dict)
    model.eval()

    test_dataset = get_dataset(args.eval_dataset or args.dataset, "test", precision=args.precision)
    test_dataset = Subset(test_dataset, list(range(0, len(test_dataset), args.dataset_skip)))
    test_loader = get_dataloader(test_dataset, False, args.batch_size, args.num_workers)

    if args.temperature_scale:
        val_dataset = get_dataset(args.dataset, "train_val", precision=args.precision)
        val_loader = get_dataloader(val_dataset, False, args.batch_size, args.num_workers)
        val_logits = torch.zeros((len(val_dataset), get_num_labels(args.dataset)))
        val_labels = torch.zeros(len(val_dataset), dtype=torch.long)
        for i, (x, y) in tqdm(enumerate(val_loader), total=len(val_loader)):
            x, y = x.to(args.device), y.to(args.device)
            if args.adv_temperature_scale:
                x = pgd_attack(model, x, y, eps=args.eps)
            lower, upper = i * args.batch_size, (i + 1) * args.batch_size
            val_logits[lower:upper] = model.forecast(model.forward(x)).logits.data
            val_labels[lower:upper] = y.data
        temperature = temperature_scale(val_labels, val_logits)
        print(f"Optimal T: {temperature:.4f}")
    else:
        temperature = 1.0

    snapshot = {
        "logits": np.zeros((len(test_dataset), get_num_labels(args.dataset))),
        "logits_adv": np.zeros((len(test_dataset), get_num_labels(args.dataset))),
        "p": np.zeros((len(test_dataset), get_num_labels(args.dataset))),
        "p_adv": np.zeros((len(test_dataset), get_num_labels(args.dataset))),
        "q": np.zeros((len(test_dataset), get_num_labels(args.dataset))),
        "cams": None,
    }

    for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):

        x, y = x.to(args.device), y.to(args.device)
        x_adv = pgd_attack(model, x, y, eps=args.eps)
        assert torch.all((x - x_adv).abs().norm(p=float("inf"), dim=(1, 2, 3)) <= args.eps + 0.01)

        preds = model.forecast(model.forward(x) / temperature)
        preds_adv = model.forecast(model.forward(x_adv) / temperature)
        cam = model.class_activation_map(x, torch.argmax(preds.probs, dim=1))

        if snapshot["cams"] is None:
            snapshot["cams"] = np.zeros((len(test_dataset), cam.shape[1], cam.shape[2]))

        lower, upper = i * args.batch_size, (i + 1) * args.batch_size
        snapshot["logits"][lower:upper, :] = preds.logits.data.cpu().numpy()
        snapshot["logits_adv"][lower:upper, :] = preds_adv.logits.data.cpu().numpy()
        snapshot["p"][lower:upper, :] = preds.probs.data.cpu().numpy()
        snapshot["p_adv"][lower:upper, :] = preds_adv.probs.data.cpu().numpy()
        snapshot["q"][lower:upper] = np.eye(get_num_labels(args.dataset))[y.data.cpu().numpy()]
        snapshot["cams"][lower:upper] = cam.data.cpu().numpy()

    save_path = f"{args.output_dir}/{args.experiment_name}"
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    for k, v in snapshot.items():
        np.save(f"{save_path}/{k}.npy", v)
