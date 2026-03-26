#!/usr/bin/env python
import logging
import os
import math
import shutil
from pathlib import Path
from argparse import ArgumentParser

import torch
from hydra.utils import instantiate
from hydra import initialize_config_dir, compose

# Fix for PyTorch 2.6+ weights_only default change
import numpy
torch.serialization.add_safe_globals([
    numpy.core.multiarray._reconstruct,
    numpy.ndarray,
    numpy.dtype,
    numpy.dtypes.UInt32DType,
])


def gradient_accumulation_steps(batch_size: int, total_batch_size: int) -> int:
    """
    total_batch_size = per_device_batch_size * world_size * grad_accum
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_per_step = batch_size * world_size
    return min(1, math.ceil(total_batch_size / effective_batch_per_step))


parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to the experiment config")
parser.add_argument("--log_level", type=int, default=logging.INFO, help="log level")
parser.add_argument("--output_dir", type=str, default=None, help="override output directory")


def main():
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=args.log_level,
    )
    logger = logging.getLogger()

    experiment_config_path = Path(args.config).expanduser().absolute()

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(torch.cuda.device_count())
        )

    # Distributed env: these are set by accelerate launch / torchrun
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")
    logger.info(f"RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")

    # Pin each process to its own GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    with initialize_config_dir(str(experiment_config_path.parents[0]), version_base=None):
        experiment_config = compose(config_name=experiment_config_path.name)

    if args.output_dir is not None:
        experiment_config.output_dir = args.output_dir

    output_dir = experiment_config.output_dir
    logger.info(f"Output directory: {output_dir}")

    # only rank 0 writes files
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(output_dir, "config.yaml"))

    trainer_config = experiment_config.trainer.copy()
    trainer = instantiate(trainer_config)

    resume_from_checkpoint = experiment_config.get("resume_from_checkpoint", None)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.evaluate()


if __name__ == "__main__":
    main()