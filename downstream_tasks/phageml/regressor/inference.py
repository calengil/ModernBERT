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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    visualize_trueFn_vs_predictedFn(targets, preds, args.out, args.task, args.epoch, mse, corr)

if __name__ == "__main__":
    main()