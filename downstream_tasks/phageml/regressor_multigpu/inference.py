import os
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import defaultdict, Counter

# --- PyTorch 2.6 safe globals (на случай, если где-то torch.load(weights_only=True))
import numpy as _np
import numpy.core.multiarray as _ncm
torch.serialization.add_safe_globals([_ncm._reconstruct, _np.ndarray, _np.dtype])
if hasattr(_ncm, "scalar"):
    torch.serialization.add_safe_globals([_ncm.scalar])


parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to hydra config yaml")
parser.add_argument("--checkpoint", type=str, required=True, help="path to HF checkpoint dir (checkpoint-XXXX/)")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--epoch", type=int, required=True)
parser.add_argument("--log_level", type=int, default=logging.INFO)
parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
parser.add_argument("--scatter", action='store_true')
parser.add_argument("--fraction", action='store_true')

@torch.no_grad()
def collect_val_predictions(trainer, device: torch.device):
    model = trainer.model
    model.eval()

    dl = trainer.get_eval_dataloader()
    preds_all = []
    targets_all = []

    for batch in dl:
        batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

        out = model(
            input_ids=batch_t["input_ids"],
            attention_mask=batch_t["attention_mask"],
            targets=batch_t.get("targets", None),
        )

        p = out.predicts
        # predicts: [B, L, 1] -> CLS -> [B]
        if p.dim() == 3:
            p = p[:, 0, 0]
        elif p.dim() == 2:
            p = p[:, 0]
        p = p.detach().cpu().numpy()
        preds_all.append(p)

        # targets: scalar per sample -> [B]
        if "targets" in batch_t:
            t = batch_t["targets"].detach().cpu().view(-1).numpy()
            targets_all.append(t)

    preds_all = np.concatenate(preds_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0) if targets_all else None

    return preds_all, targets_all


def compute_mse(preds, targets):
    preds_t = torch.as_tensor(preds, dtype=torch.float32)
    targets_t = torch.as_tensor(targets, dtype=torch.float32)
    mse_fn = nn.MSELoss()
    mse = mse_fn(preds_t, targets_t)
    return float(mse.item())


def visualize_trueFn_vs_predictedFn(x_coords, y_coords, save_path, task, epoch, mse, corr):

    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords, alpha=0.7)
    plt.xlabel('True Fn transformed')
    plt.ylabel('Predicted Fn transformed')
    plt.title(f'{task}\nMSE={mse:.6f} | r={corr:.4f} | steps{epoch}')
    plt.plot([min(x_coords), max(x_coords)], [min(x_coords), max(x_coords)], color='red', linestyle='--')
    plt.grid(True, alpha=0.3)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(f"{save_path}-scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_positive_fraction_vs_pred_threshold(
    targets,
    preds,
    save_path,
    task,
    epoch,
    target_positive_threshold=0.5,
    num_thresholds=200,
):
    targets = np.asarray(targets).reshape(-1)
    preds = np.asarray(preds).reshape(-1)

    if len(targets) != len(preds):
        raise ValueError(
            f"targets and preds must have the same length, got {len(targets)} and {len(preds)}"
        )

    valid_mask = np.isfinite(targets) & np.isfinite(preds)
    targets = targets[valid_mask]
    preds = preds[valid_mask]

    if len(targets) == 0:
        raise ValueError("No valid points left after removing NaN/Inf values.")

    thresholds = np.linspace(0.0, 1.0, num_thresholds, dtype=np.float32)

    total_n = len(targets)
    positive_fraction_values = []
    selected_fraction_values = []
    positive_kept_values = []
    selected_n_values = []

    for thr in thresholds:
        keep_mask = preds >= thr
        selected_n = int(np.sum(keep_mask))
        selected_n_values.append(selected_n)

        selected_fraction = selected_n / total_n
        selected_fraction_values.append(selected_fraction)

        if selected_n == 0:
            positive_kept = 0
            positive_fraction = np.nan
        else:
            positive_kept = int(np.sum(keep_mask & (targets > target_positive_threshold)))
            positive_fraction = positive_kept / selected_n

        positive_kept_values.append(positive_kept)
        positive_fraction_values.append(positive_fraction)

    positive_fraction_values = np.asarray(positive_fraction_values, dtype=np.float32)
    selected_fraction_values = np.asarray(selected_fraction_values, dtype=np.float32)
    positive_kept_values = np.asarray(positive_kept_values, dtype=np.int64)
    selected_n_values = np.asarray(selected_n_values, dtype=np.int64)

    finite_mask = np.isfinite(positive_fraction_values)
    if np.any(finite_mask):
        max_y = np.nanmax(positive_fraction_values)
        best_idx_candidates = np.where(np.isclose(positive_fraction_values, max_y))[0]
        #best_idx = int(best_idx_candidates[-1])  # rightmost maximum
        best_idx = int(best_idx_candidates[0])  # leftmost maximum

        best_positive_kept = int(positive_kept_values[best_idx])
        best_selected_n = int(selected_n_values[best_idx])

        positive_curve_label = (
            f"True Fn > {target_positive_threshold} among selected "
            f"(max: positive_kept={best_positive_kept}, "
            f"selected_n={best_selected_n}, total_n={int(total_n)})"
        )
    else:
        positive_curve_label = f"True Fn > {target_positive_threshold} among selected"

    plt.figure(figsize=(8, 6))
    plt.plot(
        thresholds,
        positive_fraction_values,
        linewidth=2,
        label=positive_curve_label,
    )
    plt.plot(
        thresholds,
        selected_fraction_values,
        linewidth=2,
        color='red',
        label="selected_n / total_n",
    )

    plt.xlabel("Pred Fn threshold")
    plt.ylabel("Fraction")
    plt.title(f"{task}\nsteps{epoch}")
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}-fraction_thresh.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=args.log_level)
    logger = logging.getLogger("predict")

    cfg_path = Path(args.config).expanduser().absolute()
    with initialize_config_dir(str(cfg_path.parent)):
        cfg = compose(config_name=cfg_path.name)

    # instantiate trainer (model + datasets + args)
    trainer = instantiate(cfg.trainer)

    # move model to device
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    trainer.model.to(device)

    ckpt_dir = Path(args.checkpoint).expanduser().absolute()
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"--checkpoint must be a directory like checkpoint-XXXX/, got: {ckpt_dir}")

    # HF-native loading of checkpoint directory
    logger.info(f"Loading HF checkpoint dir: {ckpt_dir}")
    trainer._load_from_checkpoint(str(ckpt_dir))

    # collect predictions on eval set
    preds, targets = collect_val_predictions(trainer, device)

    mse = compute_mse(preds, targets)
    logger.info(f"Validation MSE: {mse:.6f}")

    # === Pearson correlation (r) ===
    corr = np.corrcoef(targets, preds)[0, 1]
    #print(len(preds))
    #print(len(targets))
    #raise
    if args.scatter:
        visualize_trueFn_vs_predictedFn(targets, preds, args.out, args.task, args.epoch, mse, corr)

    if args.fraction:
        visualize_positive_fraction_vs_pred_threshold(
            targets=targets,
            preds=preds,
            save_path=args.out,
            task=args.task,
            epoch=args.epoch,
        )


if __name__ == "__main__":
    main()