#!/usr/bin/env python
# -*- coding: utf-8 -*-

# train_modernGENA_regression_no_collator.py
# ModernGENA (BERT-style MLM backbone) -> CLS (last hidden) -> regression head -> scalar
# No Horovod. No PadCollator. Padding/truncation to 1024 is done at tokenization time.

import os
import sys
import math
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup


# =========================
# ModernGENA loader (your version)
# =========================
def build_modergena_model(checkpoint_filepath, modernbert_distr_path, name, random_init):
    from omegaconf import DictConfig
    from omegaconf import OmegaConf as om
    from src import flex_bert as flex_bert_module
    from src import hf_bert as hf_bert_module
    from src import mosaic_bert as mosaic_bert_module
    from composer.utils.checkpoint import _ensure_valid_checkpoint

    def build_model(cfg: DictConfig):
        if cfg.name == "hf_bert":
            return hf_bert_module.create_hf_bert_mlm(
                pretrained_model_name=cfg.pretrained_model_name,
                use_pretrained=cfg.get("use_pretrained", None),
                model_config=cfg.get("model_config", None),
                tokenizer_name=cfg.get("tokenizer_name", None),
                gradient_checkpointing=cfg.get("gradient_checkpointing", None),
            )
        elif cfg.name == "mosaic_bert":
            return mosaic_bert_module.create_mosaic_bert_mlm(
                pretrained_model_name=cfg.pretrained_model_name,
                pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
                model_config=cfg.get("model_config", None),
                tokenizer_name=cfg.get("tokenizer_name", None),
                gradient_checkpointing=cfg.get("gradient_checkpointing", None),
            )
        elif cfg.name == "flex_bert":
            return flex_bert_module.create_flex_bert_mlm(
                pretrained_model_name=cfg.pretrained_model_name,
                pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
                model_config=cfg.get("model_config", None),
                tokenizer_name=cfg.get("tokenizer_name", None),
                gradient_checkpointing=cfg.get("gradient_checkpointing", None),
                recompute_metric_loss=cfg.get("recompute_metric_loss", False),
                disable_train_metrics=cfg.get("disable_train_metrics", False),
            )
        else:
            raise ValueError(f"Not sure how to build model with name={cfg.name}")

    cpt_dir = os.path.dirname(checkpoint_filepath)
    cfg_path = os.path.join(cpt_dir, f"{name}.txt")
    yaml_cfg = om.load(cfg_path)
    model = build_model(yaml_cfg.model)

    if not random_init:
        print(f"Loading checkpoint from {checkpoint_filepath}")
        checkpoint_filepath = Path(checkpoint_filepath)
        assert checkpoint_filepath.exists(), f"Checkpoint {checkpoint_filepath} does not exist"

        state = torch.load(_ensure_valid_checkpoint(checkpoint_filepath), map_location="cpu", weights_only=False)
        state_dict = state.get("state", {})
        model_state = state_dict.get("model", {})
        assert len(model_state) > 0, "Model state is empty, please check the checkpoint and checkpoint path"
        model.load_state_dict(model_state)

    return model


def load_model_and_tokenizer(
    cpt_path: str,
    tokenizer_path: str,
    modernbert_distr_path=None,
    name: str = "cfg.yaml",
    random_init: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = build_modergena_model(cpt_path, modernbert_distr_path, name, random_init)
    return model, tokenizer


# =========================
# Example Dataset wrapper (optional)
# If your dataset is already tokenized and padded to 1024, you can skip this.
# =========================

class TokenizedRegressionDataset(Dataset):
    def __init__(self, sequences, targets, tokenizer, max_length=1024,
                 y_min=1e-2, y_max=1e2, alpha=2.0):
        assert len(sequences) == len(targets)
        self.seqs = sequences
        self.y = targets
        self.tok = tokenizer
        self.max_length = max_length

        # transform params
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.alpha = float(alpha)

    def __len__(self):
        return len(self.seqs)

    def _transform_y(self, y: float) -> torch.Tensor:
        y = min(max(y, self.y_min), self.y_max)
        logy = math.log10(y)  # scalar float in [-2, 2]
        z = self.alpha * logy

        return torch.sigmoid(torch.tensor(z, dtype=torch.float32))

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        y_raw = float(self.y[idx])

        enc = self.tok(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = self._transform_y(y_raw)  # scalar in (0, 1)
        return item

# =========================
# Model: ModernGENA backbone -> CLS -> regression head
# =========================
class ModernGENARegressor(nn.Module):
    def __init__(self, backbone_mlm: nn.Module, hidden_size: int, head_hidden=512, head_dropout=0.1):
        super().__init__()
        self.backbone = backbone_mlm
        self.reg_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.backbone(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]  # [B, 1024, H]
        cls = last_hidden[:, 0, :]               # [B, H]
        pred = self.reg_head(cls).squeeze(-1)    # [B]
        return pred


# =========================
# Metrics
# =========================
@torch.no_grad()
def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    diff = preds - labels
    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()

    ss_res = (diff ** 2).sum()
    ss_tot = ((labels - labels.mean()) ** 2).sum()
    r2 = (1.0 - (ss_res / ss_tot)).item() if ss_tot.item() > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# =========================
# Train / Eval
# =========================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    loss_fn: nn.Module,
    grad_clip: float = 1.0,
    log_every: int = 50,
) -> float:
    model.train()
    total_loss, n = 0.0, 0

    for step, batch in enumerate(loader, start=1):
        batch = move_to_device(batch, device)

        labels = batch["labels"].float().view(-1)
        inputs = {k: v for k, v in batch.items() if k != "labels"}

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            preds = model(inputs)
            loss = loss_fn(preds, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs

        if log_every and step % log_every == 0:
            print(f"  step {step}/{len(loader)}  train_loss={total_loss/max(n,1):.6f}")

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = move_to_device(batch, device)

        labels = batch["labels"].float().view(-1)
        inputs = {k: v for k, v in batch.items() if k != "labels"}

        preds = model(inputs)
        loss = loss_fn(preds, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs

        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


# =========================
# Main training runner
# =========================
def run_training(
    args,
    train_dataset: Dataset,
    val_dataset: Dataset,
):
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # Make ModernBERT repo importable (so `from src import ...` works)
    repo_path = Path(args.modernbert_repo_path)
    sys.path.insert(0, str(repo_path))

    # Load pretrained ModernGENA MLM backbone + tokenizer
    backbone_mlm, tokenizer = load_model_and_tokenizer(
        cpt_path=args.pretrain_checkpoint,
        tokenizer_path=args.tokenizer_path,
        modernbert_distr_path=args.modernbert_distr_path,
        name=args.cfg_name,
        random_init=args.random_init,
    )

    # BERT-style hidden size
    hidden_size = int(backbone_mlm.config.hidden_size)

    model = ModernGENARegressor(
        backbone_mlm=backbone_mlm,
        hidden_size=hidden_size,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
    ).to(device)

    # IMPORTANT: dataset must already be padded/truncated to 1024 in __getitem__
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Optimizer with different LR for backbone vs head
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_enc},
            {"params": model.reg_head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fn = nn.HuberLoss(delta=args.huber_delta) if args.loss == "huber" else nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.no_fp16))

    best_val = math.inf
    best_path = os.path.join(args.out_dir, "model_best.pt")

    for ep in range(1, args.epochs + 1):
        print(f"\nEpoch {ep}/{args.epochs}")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            loss_fn=loss_fn,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
        )
        print(f"train_loss={train_loss:.6f}")

        val_metrics = evaluate(model=model, loader=val_loader, device=device, loss_fn=loss_fn)
        print(
            f"val_loss={val_metrics['loss']:.6f}  "
            f"MAE={val_metrics['mae']:.6f}  RMSE={val_metrics['rmse']:.6f}  R2={val_metrics['r2']:.6f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {"state_dict": model.state_dict(), "best_val_loss": best_val, "args": vars(args)},
                best_path,
            )
            print(f"saved best -> {best_path}")

    print(f"\nBest val loss: {best_val:.6f}")
    return best_path


def parse_args():
    p = argparse.ArgumentParser()

    # ModernBERT repo (so `from src import ...` works)
    p.add_argument("--modernbert_repo_path", type=str, required=True)
    p.add_argument("--modernbert_distr_path", type=str, default=None)

    # Pretrain checkpoint + tokenizer
    p.add_argument("--pretrain_checkpoint", type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument("--cfg_name", type=str, default="cfg.yaml")
    p.add_argument("--random_init", action="store_true")

    # Training
    p.add_argument("--out_dir", type=str, default="./runs_modernGENA_reg")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # Optim / sched
    p.add_argument("--lr_enc", type=float, default=2e-5)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Head
    p.add_argument("--head_hidden", type=int, default=512)
    p.add_argument("--head_dropout", type=float, default=0.1)

    # Loss / AMP
    p.add_argument("--loss", type=str, choices=["huber", "mse"], default="mse")
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--no_fp16", action="store_true")

    # Logging
    p.add_argument("--log_every", type=int, default=50)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()


    raw_train = ...
    raw_val = ...
    backbone_tmp, tok = load_model_and_tokenizer(args.pretrain_checkpoint, args.tokenizer_path, args.modernbert_distr_path, args.cfg_name, args.random_init)
    train_dataset = TokenizedRegressionDataset(raw_train, tok, max_length=1024)
    val_dataset   = TokenizedRegressionDataset(raw_val, tok, max_length=1024)
    best = run_training(args, train_dataset, val_dataset)
    print("BEST:", best)