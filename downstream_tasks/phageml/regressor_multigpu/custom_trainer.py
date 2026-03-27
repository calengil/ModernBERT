import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback

from torchmetrics import Metric
from torchmetrics.classification import BinaryAveragePrecision

import math

import shutil
from pathlib import Path
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class NamedMean(Metric):
    def __init__(self, cls_name: str, **kwargs):
        super().__init__(**kwargs)
        self.cls_name = cls_name
        self.log_name = f"{cls_name}"
        # Keep distributed reduction as you had it; compute() will sync across ranks.
        self.add_state("sum", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, inputs, model_outputs):
        v = model_outputs[self.cls_name]
        if isinstance(v, torch.Tensor):
            v = v.detach().float()
        self.sum += v
        self.count += 1.0

    def compute(self):
        return self.sum / torch.clamp_min(self.count, 1.0)


class NamedBinaryAveragePrecision(BinaryAveragePrecision):
    def __init__(self, cls_name: str, **kwargs):
        super().__init__(**kwargs)
        self.cls_name = cls_name
        self.log_name = f"PRAUC_{cls_name}"

    def update(self, inputs, model_outputs):
        targets = inputs["targets"]
        predicts = model_outputs["logits"]

        if isinstance(predicts, torch.Tensor):
            predicts = predicts.detach()
        # targets is a dict of tensors; keep them as tensors for torchmetrics
        class_index = 0
        updated = False

        for class_name in ["tss", "polya"]:
            for strand in ["+", "-"]:
                cls_name = f"primary_{class_name}_{strand}"
                if self.cls_name == cls_name:
                    X = predicts[:, :, class_index]
                    Y = (targets[cls_name] > 0.5)
                    if isinstance(Y, torch.Tensor):
                        Y = Y.detach()
                    super().update(X, Y)
                    updated = True
                    break
                class_index += 1
            if updated:
                break

        if not updated and predicts.shape[-1] == 6:
            for lidx, strand in enumerate(["+", "-"]):
                cls_name = f"intragenic_regions_{strand}"
                if self.cls_name == cls_name:
                    X = predicts[:, :, 4 + lidx]
                    Y = (targets[cls_name] > 0.5)
                    if isinstance(Y, torch.Tensor):
                        Y = Y.detach()
                    super().update(X, Y)
                    updated = True
                    break

        if not updated:
            raise ValueError(f"Class name {self.cls_name} not found in targets: \n{targets.keys()}")


class DetectTrainStepStart(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        control.is_in_train_step = True
        return control


class LogTrainMetricsCallback(TrainerCallback):
    """
    CRITICAL FIX:
    - ALL ranks must call metric.compute()/reset() on the same steps,
      otherwise TorchMetrics will hang in distributed sync collectives. :contentReference[oaicite:1]{index=1}
    - Only rank 0 logs.
    """
    def on_step_end(self, args, state, control, **kwargs):
        trainer = getattr(state, "trainer_instance", None)
        if trainer is None:
            control.is_in_train_step = False
            return control

        if trainer.log_train_metrics is None:
            control.is_in_train_step = False
            state.trainer_instance = None
            return control

        if state.global_step % trainer.log_train_metrics != 0:
            control.is_in_train_step = False
            state.trainer_instance = None
            return control

        # Compute+reset on ALL ranks to avoid deadlock
        results = {}
        for m in trainer.train_metrics:
            val = m.compute()
            # convert to python scalar
            if isinstance(val, torch.Tensor):
                val_item = val.detach().float().cpu().item()
            else:
                val_item = float(val)
            results[m.log_name] = val_item
            m.reset()

        # Only rank 0 logs
        if state.is_world_process_zero:
            original_should_log_state = control.should_log
            trainer.log({f"train_{k}": v for k, v in results.items()})
            control.should_log = original_should_log_state

        control.is_in_train_step = False
        state.trainer_instance = None
        return control


class DataloaderWithEpochReseed(DataLoader):
    def set_epoch(self, epoch):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "reseed_epoch"):
            self.dataset.reseed_epoch(epoch + 1)


class DistributedWeightedSampler(Sampler):
    def __init__(
        self,
        weights: torch.Tensor,
        num_replicas: int,
        rank: int,
        replacement: bool = True,
        seed: int = 0,
    ):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0

        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas - 1}], got {self.rank}")

        self.num_samples = math.ceil(len(self.weights) / self.num_replicas)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * self.num_replicas + self.rank)

        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
            generator=g,
        )

        return iter(indices.tolist())

from omegaconf import OmegaConf
class CustomTrainer(Trainer):
    """
    Minimal trainer:
    - Forces DistributedSampler when WORLD_SIZE > 1 (so steps/epoch scales with GPUs)
    - Uses worker_init_fn
    - Removes prefetch_factor/persistent_workers extras
    - Fixes TorchMetrics logging deadlock under DDP
    """

    def __init__(self, *args, **kwargs):
        self.log_train_metrics = kwargs.pop("log_train_metrics", None)
        self.train_metrics = kwargs.pop("train_metrics", [])

        #callbacks = kwargs.pop("callbacks", [])
        #if self.log_train_metrics is not None:
        #    callbacks.append(LogTrainMetricsCallback())
        #    callbacks.append(DetectTrainStepStart())
        #kwargs["callbacks"] = callbacks

        callbacks = kwargs.pop("callbacks", None)
        callbacks = [] if callbacks is None else list(callbacks)

        if self.log_train_metrics is not None:
            callbacks.append(LogTrainMetricsCallback())
            callbacks.append(DetectTrainStepStart())

        kwargs["callbacks"] = callbacks


        super().__init__(*args, **kwargs)

        if OmegaConf.is_config(self.eval_dataset):
            self.eval_dataset = OmegaConf.to_object(self.eval_dataset)

        for cb in self.callback_handler.callbacks:
            if hasattr(cb, "set_trainer"):
                cb.set_trainer(self)

        if self.log_train_metrics is not None:
            self.control.is_in_train_step = False
            for m in self.train_metrics:
                m.to(self.args.device)

    @staticmethod
    def _env_world():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        return world_size, rank

    def _get_worker_init_fn(self):
        from downstream_tasks.phageml.regressor_multigpu.simple_test_dataset import worker_init_fn #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return worker_init_fn

    def _get_train_sampler(self):
        if self.train_dataset is None:
            return None

        world_size, rank = self._env_world()
        if world_size > 1:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
                seed=self.args.seed,
            )
        return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = self._get_train_sampler()

        return DataloaderWithEpochReseed(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=self._get_worker_init_fn(),
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if isinstance(eval_dataset, str):
            if not isinstance(self.eval_dataset, dict):
                raise ValueError("String eval_dataset is only supported when self.eval_dataset is a dict.")
            eval_dataset = self.eval_dataset[eval_dataset]
        elif eval_dataset is None:
            eval_dataset = self.eval_dataset

        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        world_size, rank = self._env_world()
        if world_size > 1:
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
                seed=self.args.seed,
            )
        else:
            eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataloaderWithEpochReseed(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=self._get_worker_init_fn(),
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model=model, inputs=inputs, return_outputs=True, **kwargs)

        if self.log_train_metrics is not None and getattr(self.control, "is_in_train_step", False):
            # Make trainer accessible to the callback in this process
            self.state.trainer_instance = self
            for m in self.train_metrics:
                m.update(inputs, outputs)

        return (loss, outputs) if return_outputs else loss
    


class CustomRegressionTrainer(CustomTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_targets = "targets" in inputs and inputs["targets"] is not None

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    targets=inputs.get("targets", None),
                )

        loss = None
        if has_targets and hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss.detach()

        if prediction_loss_only:
            return (loss, None, None)

        preds = outputs.predicts
        # exactly like in your inference code
        if preds.dim() == 3:
            preds = preds[:, 0, 0]
        elif preds.dim() == 2:
            preds = preds[:, 0]

        preds = preds.detach()

        labels = None
        if has_targets:
            labels = inputs["targets"]
            if isinstance(labels, torch.Tensor):
                labels = labels.view(-1).detach()

        return (loss, preds, labels)

class CustomTrainerWeighted(CustomRegressionTrainer):
    """
    weighted_sampling задаётся из YAML:
      weighted_sampling:
        mode: bins
        bins: [0.5, 0.85, 0.95]
        weights: [1.0, 2.0, 4.0, 8.0]
    """

    def __init__(self, *args, **kwargs):
        self.weighted_sampling = kwargs.pop("weighted_sampling", None)
        super().__init__(*args, **kwargs)
        if self.weighted_sampling is None:
            raise ValueError("CustomTrainerWeighted requires `weighted_sampling` in YAML")

    def _get_train_sampler(self):
        if self.train_dataset is None:
            return None

        world_size, rank = self._env_world()
        if world_size > 1:
            return super()._get_train_sampler()

        # Single GPU: WeightedRandomSampler
        weights = self._build_weights_bins()
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.train_dataset),
            replacement=True,
        )

    def _build_weights_bins(self) -> torch.DoubleTensor:
        cfg = self.weighted_sampling
        if cfg.get("mode", "bins") != "bins":
            raise ValueError("Only mode='bins' is implemented")

        bins = [float(x) for x in cfg.get("bins", [0.5, 0.85, 0.95])]
        wts  = [float(x) for x in cfg.get("weights", [1.0, 2.0, 4.0, 8.0])]

        if len(wts) != len(bins) + 1:
            raise ValueError("weighted_sampling.weights must have length len(bins)+1")

        ds = self.train_dataset
        n = len(ds)

        fast = getattr(ds, "get_target_value", None)
        y = torch.empty(n, dtype=torch.float64)

        for i in range(n):
            if fast is not None:
                y[i] = float(fast(i))
            else:
                item = ds[i]
                y[i] = float(item["targets"].view(-1)[0].item())


        thresholds = torch.quantile(y, torch.tensor(bins, dtype=torch.float64))

        weights = torch.empty_like(y)

        for bi in range(len(wts)):
            if bi == 0:
                mask = y <= thresholds[0]
            elif bi == len(wts) - 1:
                mask = y > thresholds[-1]
            else:
                mask = (y > thresholds[bi - 1]) & (y <= thresholds[bi])
            weights[mask] = wts[bi]

        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if float(weights.sum().item()) == 0.0:
            weights = torch.ones_like(weights)

        return weights.to(torch.double)

import torch
from torch.utils.data import WeightedRandomSampler

class CustomTrainerWeightedBalancedBins(CustomRegressionTrainer):
    def __init__(self, *args, **kwargs):
        self.weighted_sampling = kwargs.pop("weighted_sampling", None)
        super().__init__(*args, **kwargs)
        if self.weighted_sampling is None:
            raise ValueError("CustomTrainerWeightedBalancedBins requires `weighted_sampling` in YAML")

    def _get_train_sampler(self):
        if self.train_dataset is None:
            return None

        world_size, rank = self._env_world()
        weights = self._build_balanced_bin_weights()

        if world_size > 1:
            return DistributedWeightedSampler(
                weights=weights,
                num_replicas=world_size,
                rank=rank,
                replacement=True,
                seed=self.args.seed,
            )

        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.train_dataset),
            replacement=True,
        )

    def _build_balanced_bin_weights(self) -> torch.DoubleTensor:
        cfg = self.weighted_sampling
        mode = cfg.get("mode", "balanced_bins")
        if mode != "balanced_bins":
            raise ValueError("CustomTrainerWeightedBalancedBins supports only mode='balanced_bins'")

        edges = cfg.get("bins_edges", None)
        if edges is None or len(edges) == 0:
            raise ValueError("weighted_sampling.bins_edges must be a non-empty list of increasing floats")

        edges = [float(x) for x in edges]
        if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
            raise ValueError("bins_edges must be strictly increasing")

        boost_bin = cfg.get("boost_bin", None)
        boost = float(cfg.get("boost", 1.0))
        empty_policy = cfg.get("empty_bin_policy", "skip")  # skip or error

        ds = self.train_dataset
        n = len(ds)

        fast = getattr(ds, "get_target_value", None)
        y = torch.empty(n, dtype=torch.float64)
        for i in range(n):
            if fast is not None:
                y[i] = float(fast(i))
            else:
                item = ds[i]
                y[i] = float(item["targets"].view(-1)[0].item())

        # Бины: [-inf, e0], (e0,e1], ... , (e_last, +inf)
        edges_t = torch.tensor(edges, dtype=torch.float64)
        # bucketize returns bin index in [0..len(edges)]
        # with right=True: values equal to edge go to the right bin (<= edge)
        bin_idx = torch.bucketize(y, edges_t, right=True)  # shape [n], values 0..B-1 where B=len(edges)+1
        B = len(edges) + 1

        # count per bin
        counts = torch.bincount(bin_idx, minlength=B).to(torch.float64)

        # Handle empty bins
        nonempty = counts > 0
        if not torch.all(nonempty):
            empty_bins = torch.where(~nonempty)[0].tolist()
            if empty_policy == "error":
                raise ValueError(f"Empty bins found: {empty_bins}. Adjust bins_edges or policy.")
            # skip: we'll set their desired mass to 0 and renormalize over non-empty bins

        # Desired probability mass per bin:
        # - equal across non-empty bins
        # - optionally boosted for one bin
        desired = torch.zeros(B, dtype=torch.float64)
        nonempty_bins = torch.where(nonempty)[0]
        k = int(nonempty_bins.numel())
        if k == 0:
            # should never happen
            return torch.ones(n, dtype=torch.double)

        # start with uniform over non-empty
        desired[nonempty_bins] = 1.0 / k

        # optional boost: multiply desired mass of that bin, then renormalize
        if boost_bin is not None:
            boost_bin = int(boost_bin)
            if boost_bin < 0 or boost_bin >= B:
                raise ValueError(f"boost_bin must be in [0, {B-1}], got {boost_bin}")
            if nonempty[boost_bin]:
                desired[boost_bin] = desired[boost_bin] * boost

        # renormalize desired over non-empty bins
        s = desired.sum().item()
        if s <= 0:
            # fallback
            desired[nonempty_bins] = 1.0 / k
            s = desired.sum().item()
        desired = desired / s

        # Now set per-sample weights so that bin mass matches desired:
        # if bin b has count c_b and we want total mass desired_b,
        # then each sample in bin b gets weight proportional to desired_b / c_b.
        per_bin_weight = torch.zeros(B, dtype=torch.float64)
        per_bin_weight[nonempty_bins] = desired[nonempty_bins] / counts[nonempty_bins]

        weights = per_bin_weight[bin_idx]
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if float(weights.sum().item()) == 0.0:
            weights = torch.ones_like(weights)

        return weights.to(torch.double)
    


class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, every_n_epochs: int, extra_dir: str = "periodic"):
        self.every_n_epochs = int(every_n_epochs)
        self.extra_dir = extra_dir
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.trainer is None or state.epoch is None:
            return control

        epoch_int = int(round(state.epoch))
        is_exact_epoch = abs(state.epoch - epoch_int) < 1e-8

        if not is_exact_epoch or epoch_int == 0 or epoch_int % self.every_n_epochs != 0:
            return control

        original_output_dir = self.trainer.args.output_dir
        original_save_total_limit = self.trainer.args.save_total_limit

        try:
            # save directly into a separate folder
            self.trainer.args.output_dir = os.path.join(original_output_dir, self.extra_dir)
            # do not rotate periodic checkpoints with the main save_total_limit
            self.trainer.args.save_total_limit = None
            self.trainer._save_checkpoint(model, trial=None)
        finally:
            self.trainer.args.output_dir = original_output_dir
            self.trainer.args.save_total_limit = original_save_total_limit

        return control