import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from segnet.model import SegNet

dir_path = os.path.dirname(os.path.realpath(__file__))


def train(model: SegNet,
          train_data_loader: DataLoader,
          validation_data_loader: DataLoader,
          num_labels: int,
          epochs: int = 100,
          learning_rate: float = 3e-4,
          device: str = 'cpu') -> SegNet:
    optimizer = Adam(model.train_parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss()

    epoch_losses = []
    for epoch in range(1, epochs + 1):
        train_epoch_loss = __train(model, train_data_loader, num_labels, optimizer, loss_fn, device)
        validation_data_loss = __validate(model, validation_data_loader, num_labels, loss_fn, device)
        print(f'[{epoch}/{epochs}]\tTrain epoch loss: {train_epoch_loss}\tValidation epoch loss: {validation_data_loss}')

        torch.save(model.state_dict(), os.path.join(dir_path, 'weights', f'seg_net_weights_{epoch}.pth'))

        if epoch > 3 and np.mean(epoch_losses[-3:]) < validation_data_loss:
            print('Overfitting detected. Stopping training...')
            return model

        epoch_losses.append(validation_data_loss)
    return model


def __train(model: SegNet,
            data_loader: DataLoader,
            num_labels: int,
            optimizer,
            loss_fn,
            device: str = 'cpu') -> float:
    model = model.train()
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
                'Epoch loss': epoch_loss / batch_idx
            })

        return epoch_loss / batch_idx


def __validate(model: SegNet,
               data_loader: DataLoader,
               num_labels: int,
               loss_fn,
               device: str = 'cpu') -> float:
    model = model.eval()
    with tqdm(data_loader, total=len(data_loader)) as samples:
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(samples, start=1):
            imgs, targets = batch[0].to(device), batch[1].to(device)

            outputs = model(imgs)
            outputs = outputs.permute(0, 2, 3, 1).reshape(-1, num_labels)
            targets = targets.reshape(-1)

            loss = loss_fn(outputs, targets)

            epoch_loss += loss.item()

            samples.set_postfix({
                'Epoch loss': epoch_loss / batch_idx
            })

        return epoch_loss / batch_idx
