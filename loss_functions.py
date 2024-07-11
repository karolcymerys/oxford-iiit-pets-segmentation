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


class DiceLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.iou = IOULoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 1 - self.iou(logits, targets)


class IOULoss(torch.nn.Module):
    def __init__(self, epsilon: float = 1e-7) -> None:
        super(IOULoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bs = logits.shape[0]
        num_classes = logits.shape[1]

        preds = logits.log_softmax(dim=1).exp()

        predicted = preds.view(bs, num_classes, -1)
        true_positive = torch.nn.functional.one_hot(targets.view(bs, -1).long(), num_classes).permute(0, 2, 1)

        intersection = (predicted*true_positive).sum()
        union = (predicted + true_positive).sum()

        return (2.0 * intersection + self.epsilon) / (union + self.epsilon)
