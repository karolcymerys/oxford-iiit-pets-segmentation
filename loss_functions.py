import torch
from torch.nn import CrossEntropyLoss


class CustomCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_labels: int) -> None:
        super(CustomCrossEntropyLoss, self).__init__()
        self.loss_fn = CrossEntropyLoss()
        self.num_labels = num_labels

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds_flat = logits.permute(0, 2, 3, 1).reshape(-1, self.num_labels)
        targets_flat = targets.reshape(-1)
        return self.loss_fn(preds_flat, targets_flat)
