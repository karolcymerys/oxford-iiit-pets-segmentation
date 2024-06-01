import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from segnet.model import SegNet


def train(model: SegNet,
          data_loader: DataLoader,
          num_labels: int,
          epochs: int = 10,
          learning_rate: float = 1e-3,
          device: str = 'cpu') -> SegNet:
    optimizer = Adam(model.train_parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        with tqdm(data_loader, total=len(data_loader)) as samples:
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(samples, start=1):
                imgs, targets = batch[0].to(device), batch[1].to(device)

                outputs = model(imgs)
                outputs = outputs.permute(0, 2, 3, 1).reshape(-1, num_labels)
                targets = targets.reshape(-1)

                optimizer.zero_grad()
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                samples.set_postfix({
                    'Epoch': epoch,
                    'Epoch loss': epoch_loss / batch_idx
                })

        torch.save(model.state_dict(), f'seg_net_weights_{epoch}.pth')

    return model
