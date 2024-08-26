import os

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from deeplab.model import DeeplabV3

dir_path = os.path.dirname(os.path.realpath(__file__))


def train(model: DeeplabV3,
          train_data_loader: DataLoader,
          validation_data_loader: DataLoader,
          loss_fn: torch.nn.Module,
          model_suffix: str,
          epochs: int = 100,
          learning_rate: float = 1e-2,
          device: str = 'cpu') -> DeeplabV3:
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.99, weight_decay=1e-4)

    epoch_losses = []
    for epoch in range(1, epochs + 1):
        train_epoch_loss = __train(model, train_data_loader, optimizer, loss_fn, device)
        validation_data_loss = __validate(model, validation_data_loader, loss_fn, device)
        print(f'[{epoch}/{epochs}]\tTrain epoch loss: {train_epoch_loss}\tValidation epoch loss: {validation_data_loss}')

        torch.save(model.state_dict(), os.path.join(dir_path, 'weights', f'deeplab_v3_weights_{epoch}_{model_suffix}.pth'))

        if epoch > 10 and np.mean(epoch_losses[-5:]) < validation_data_loss:
            print('Overfitting detected. Stopping training...')
            return model

        epoch_losses.append(validation_data_loss)
    return model


def __train(model: DeeplabV3,
            data_loader: DataLoader,
            optimizer,
            loss_fn,
            device: str = 'cpu') -> float:
    model = model.train()
    with tqdm(data_loader, total=len(data_loader)) as samples:
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(samples, start=1):
            imgs, targets = batch[0].to(device), batch[1].to(device)

            outputs = model(imgs)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            samples.set_postfix({
                'Epoch loss': epoch_loss / batch_idx
            })

        return epoch_loss / batch_idx


def __validate(model: DeeplabV3,
               data_loader: DataLoader,
               loss_fn,
               device: str = 'cpu') -> float:
    model = model.eval()
    with tqdm(data_loader, total=len(data_loader)) as samples:
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(samples, start=1):
            imgs, targets = batch[0].to(device), batch[1].to(device)

            outputs = model(imgs)

            loss = loss_fn(outputs, targets)

            epoch_loss += loss.item()

            samples.set_postfix({
                'Epoch loss': epoch_loss / batch_idx
            })

        return epoch_loss / batch_idx
