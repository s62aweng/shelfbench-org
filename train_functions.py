import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import torch.nn as nn
import torch.optim as optim
import logging
from omegaconf import DictConfig


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    cfg: DictConfig,
    log: logging.Logger,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    for batch_idx, (image, mask) in enumerate(
        tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")
    ):
        image = image.to(device)
        mask = mask.to(device)

        # TODO: Check if the mask really is inverted
        mask = 1 - (mask / 255)
        mask = F.one_hot(mask.long(), num_classes=2).squeeze(1).permute(0, 3, 1, 2)

        optimizer.zero_grad()

        prediction = model(image)

        loss = loss_function(prediction, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % cfg["training"].get("log_interval", 10) == 0:
            log.info(
                f"Train Epoch: {epoch+1} [{batch_idx * len(image)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

    epoch_loss = running_loss / len(train_loader)
    log.info(f"Train Epoch: {epoch+1} Average Loss: {epoch_loss:.6f}")

    return epoch_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    cfg: DictConfig,
    log: logging.Logger,
    epoch: int,
) -> float:
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for image, mask in tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            image = image.to(device)
            mask = mask.to(device)

            mask = 1 - (mask / 255)
            mask = F.one_hot(mask.long(), num_classes=2).squeeze(1).permute(0, 3, 1, 2)

            prediction = model(image)

            loss = loss_function(prediction, mask)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    log.info(f"Validation Epoch: {epoch+1} Average Loss: {val_loss:.6f}")

    return val_loss
