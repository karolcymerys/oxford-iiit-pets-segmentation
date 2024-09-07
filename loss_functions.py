import torch
from torch.nn import CrossEntropyLoss


class CrossEntropyLossWrapper(torch.nn.Module):
    def __init__(self, num_labels: int) -> None:
        super(CrossEntropyLossWrapper, self).__init__()
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

        predicted_labels = preds.view(bs, num_classes, -1)
        known_labels = torch.nn.functional.one_hot(targets.view(bs, -1).long(), num_classes).permute(0, 2, 1)

        intersection = (predicted_labels * known_labels).sum()
        union = (predicted_labels + known_labels).sum()

        return (2.0 * intersection + self.epsilon) / (union + self.epsilon)


class BCELoss(torch.nn.Module):
    def __init__(self) -> None:
        super(BCELoss, self).__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bs = logits.shape[0]
        num_classes = logits.shape[1]

        preds = logits.log_softmax(dim=1).exp().view(-1)
        targets = torch.nn.functional.one_hot(targets.view(bs, -1).long(), num_classes).permute(0, 2, 1).reshape(-1).float()
        return torch.nn.functional.binary_cross_entropy(preds, targets, reduction='mean')


class DiceBCELoss(torch.nn.Module):
    def __init__(self) -> None:
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = BCELoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_loss(logits, targets) + self.bce_loss(logits, targets)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super(FocalLoss, self).__init__()
        self.bce_loss = BCELoss()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_loss(logits, targets)
        return self.alpha * (1 - bce_loss.exp())**self.gamma * bce_loss


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, epsilon: float = 1.0) -> None:
        super(TverskyLoss, self).__init__()
        self.bce_loss = BCELoss()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bs = logits.shape[0]
        num_classes = logits.shape[1]

        preds = logits.log_softmax(dim=1).exp()

        predicted_labels = preds.view(bs, num_classes, -1)
        known_labels = torch.nn.functional.one_hot(targets.view(bs, -1).long(), num_classes).permute(0, 2, 1)

        TP = (predicted_labels * known_labels).sum()
        FP = ((1 - known_labels) * predicted_labels).sum()
        FN = (known_labels * (1 - predicted_labels)).sum()

        return 1 - ((TP + self.epsilon) / (TP + self.alpha * FP + self.beta * FN + self.epsilon))


class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 4/3, epsilon: float = 1.0) -> None:
        super(FocalTverskyLoss, self).__init__()
        self.tversky_loss = TverskyLoss(alpha, beta, epsilon)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.tversky_loss(logits, targets) ** self.gamma
