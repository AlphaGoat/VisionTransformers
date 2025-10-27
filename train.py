"""
Base training script for our vision transformers.

Author: Peter Thomas
Date: 07 October 2025
"""
import torch
import pyrallis

from config import TrainConfig
from vision_transformers import build_model
from metrics import ObjectDetectionAnalyzer
from datasets import ObjectDetectionDataset



def train(model,
          train_dataset,
          val_dataset,
          criterion,
          optimizer,
          lr_scheduler,
          num_epochs,
          batch_size,
          device):

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Empty dictionary with metrics to monitor 
    metrics_to_monitor = {}

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, targets in train_dataloader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        lr_scheduler.step()

        # Validation loop (optional)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_dataloader:
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

@pyrallis.wrap()
def main(args):
    config = TrainConfig.from_yaml(args.config)

    # Build model and retrieve criterion
    criterion, model = build_model(**config.model)

    # Move model to device
    model.to(args.device)

    # Define optimizer, and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.optimizer.step_size, gamma=config.optimizer.gamma)

    # Load datasets
    train_dataset = ObjectDetectionDataset(config.dataset.train_path, split='train')
    val_dataset = ObjectDetectionDataset(config.dataset.val_path, split='val')

    # Start training
    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=config.training.num_epochs,
        batch_size=config.training.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a vision transformer model.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu').")
    args = parser.parse_args()

    main(args)