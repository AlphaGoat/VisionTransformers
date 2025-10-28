"""
Base training script for our vision transformers.

Author: Peter Thomas
Date: 07 October 2025
"""
import torch
import pyrallis

import metrics
from config import TrainObjDetConfig
from utils.logger import Logger
from vision_transformers import build_model
from datasets import ObjectDetectionDataset
from hooks import get_attention_weights, get_layer_statistics



def train(model,
          train_dataset,
          val_dataset,
          criterion,
          optimizer,
          logger,
          lr_scheduler,
          num_epochs,
          batch_size,
          device):

    # Initialize dataloaders, including random sampler to plot attention weight mechanisms
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_random_sampler = torch.utils.data.RandomSampler(val_dataset, replacement=False)
    random_sampler_dataloader = torch.utils.data.DataLoader(val_dataset, sampler=val_random_sampler, batch_size=1)

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
        epoch_outputs = []
        epoch_targets = []
        with torch.no_grad():
            for images, targets in val_dataloader:
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

                epoch_outputs.append([{k: v.detach().cpu().numpy() for k, v in outputs.items()}])
                epoch_targets.append([{k: v.detach().cpu().numpy() for k, v in t.items()} for t in targets])

        oda_metrics = metrics.perform_oda_evaluation(epoch_targets, epoch_outputs)

        # Get layer statistics and attention weights after epoch of training
        layer_stats = get_layer_statistics(model)
        attention_weights = get_attention_weights(model)

        random_img = next(iter(random_sampler_dataloader))
        random_img = random_img[0].to(device)

        feature_map = model.backbone(random_img.unsqueeze(0))


        # Log metrics
        logger.log_metrics(
            epoch,
            oda_metrics=oda_metrics,
            layer_stats=layer_stats,
        )

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")


@pyrallis.wrap()
def main(args):
    config = TrainObjDetConfig.from_yaml(args.config)

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

    # initialize logger
    logger = Logger(log_dir=config.logging_path)

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