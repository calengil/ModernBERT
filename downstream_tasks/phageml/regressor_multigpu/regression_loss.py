import torch
import torch.nn as nn

class CLSMSELossDict(nn.Module):
    def __init__(self, cls_pos: int = 0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cls_pos = int(cls_pos)

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor):
        if targets is None:
            raise ValueError("targets is None: cannot compute loss")

        # CLS-slice -> [B]
        if predicts.dim() == 3:
            # [B, L, 1] -> [B]
            pred = predicts[:, self.cls_pos, 0]
        elif predicts.dim() == 2:
            # [B, L] -> [B]
            pred = predicts[:, self.cls_pos]
        else:
            # [B] or [B,1] -> [B]
            pred = predicts.view(-1)

        tgt = targets.float().view(-1)

        total = self.mse(pred, tgt)

        zero = total.detach() * 0.0
        return {
            "total": total,
            "tss": zero,
            "polya": zero,
            "intragenic": zero,
        }



def compute_regression_metrics(eval_pred):
    import numpy as np

    preds = np.asarray(eval_pred.predictions).reshape(-1)
    labels = np.asarray(eval_pred.label_ids).reshape(-1)

    mse = float(np.mean((preds - labels) ** 2))

    if len(preds) < 2 or np.std(preds) == 0 or np.std(labels) == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(labels, preds)[0, 1])

    return {
        "mse": mse,
        "pearson": pearson,
    }