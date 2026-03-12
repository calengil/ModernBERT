import math
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
import h5py


# ============================================================
# DataLoader worker init (so CustomTrainer can safely open files
# in each worker process)
# ============================================================
def worker_init_fn(worker_id: int):
    """
    Initialize worker with proper file handling.

    CustomTrainer imports this function as:
        from simple_annotation_dataset import worker_init_fn

    So either:
      - put THIS file on PYTHONPATH named simple_annotation_dataset.py
      - OR keep this function here and also create a tiny
        simple_annotation_dataset.py that imports it.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return

    ds = worker_info.dataset

    def _open(ds0):
        # If dataset supports open_files(), call it
        if hasattr(ds0, "open_files"):
            ds0.open_files()
            return

        # Handle ConcatDataset / Subset recursively
        if isinstance(ds0, ConcatDataset):
            for child in ds0.datasets:
                _open(child)
            return

        if isinstance(ds0, Subset):
            _open(ds0.dataset)
            return

        # If dataset doesn't need files, do nothing
        return

    _open(ds)



def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
		"""
		Collate function for DataLoader.
		
		Args:
			batch: List of samples from __getitem__
			
		Returns:
			Batched data
		"""
		batched = {
			'input_ids': torch.stack([item['input_ids'] for item in batch]),
			'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
			'targets': {},
		}
		
		# Batch targets
		if batch[0]['targets'] is None:
			assert np.all([item['targets'] is None for item in batch]), "All targets must be None for inference"
			batched['targets'] = None
		else:
			target_keys = batch[0]['targets'].keys()
			for key in target_keys:
				batched['targets'][key] = torch.stack([item['targets'][key] for item in batch])

		return batched		

class RegressionScoringDataset(Dataset):
    """
    HDF5 structure:
      /sample_0
        attrs['seq'] = "ACGT..."
        dataset["fn_transform"] = np.array([fn], dtype=float64)
      /sample_1
        ...

    Returns dict:
      input_ids: LongTensor[1024]
      attention_mask: LongTensor[1024]
      token_type_ids: LongTensor[1024] (zeros if tokenizer doesn't provide it)
      targets: FloatTensor scalar (sigmoid(alpha*log10(clip(y))))
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        y_min: float = 1e-2,
        y_max: float = 1e2,
        alpha: float = 1.0,
        group_prefix: str = "sample_",
    ):
        self.data_path = str(data_path)
        self.tok = tokenizer
        self.max_length = int(max_length)

        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.alpha = float(alpha)
        self.group_prefix = str(group_prefix)

        # IMPORTANT: do NOT keep an open HDF5 handle across processes.
        # Each DataLoader worker should open its own handle via open_files()/worker_init_fn.
        self._h5: Optional[h5py.File] = None

        # Cache group keys (safe to do once)
        with h5py.File(self.data_path, "r") as f:
            keys = list(f.keys())
            # stable numeric sort for sample_0, sample_1, ...
            def _key_sort(k: str):
                if k.startswith(self.group_prefix):
                    tail = k.replace(self.group_prefix, "")
                    return int(tail) if tail.isdigit() else k
                return k

            self._keys: List[str] = sorted(keys, key=_key_sort)

    # -------------------------
    # File handling for workers
    # -------------------------
    def open_files(self):
        """Open HDF5 file in the current process (worker)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.data_path, "r")

    def close_files(self):
        """Close HDF5 file in the current process."""
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5 = None

    def _ensure_open(self):
        if self._h5 is None:
            # Fallback in case worker_init_fn wasn't used
            self.open_files()

    # -------------------------
    # Dataset API
    # -------------------------
    def __len__(self):
        return len(self._keys)

    def _transform_y(self, y: float) -> torch.Tensor:
        y = min(max(y, self.y_min), self.y_max)
        logy = math.log10(y)  # scalar float in [-2, 2]
        z = self.alpha * logy

        return torch.sigmoid(torch.tensor(z, dtype=torch.float32))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._ensure_open()

        key = self._keys[idx]
        grp = self._h5[key]

        seq = grp.attrs["seq"]
        if isinstance(seq, bytes):
            seq = seq.decode("utf-8")

        y_raw = float(np.asarray(grp["fn_transform"])[0])

        enc = self.tok(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items() if k in ['input_ids', 'attention_mask']}  # input_ids, attention_mask, maybe token_type_ids

        # Ensure token_type_ids exists (BERT-style)
        ###if "token_type_ids" not in item:
        ###    item["token_type_ids"] = torch.zeros(self.max_length, dtype=torch.long)

        # IMPORTANT: key name "targets" (AnnotationModel.forward expects targets)
        item["targets"] = y_raw #self._transform_y(y_raw)

        return item

    def __del__(self):
        # Best-effort close
        try:
            self.close_files()
        except Exception:
            pass