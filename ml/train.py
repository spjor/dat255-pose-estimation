import os
import time

import torch
import torchvision.models.detection
import wandb
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from ml.config import *
from ml.dataset import COCOKeypointDataset
from ml.loss import KeypointMSELoss
from ml.models.pose_model import PoseEstimationModel, get_pose_convnet, get_pose_resnet18, get_pose_resnet50, \
    get_pose_resnet101, get_pose_resnet34, get_pose_resnet152
from ml.models.resnet_backbone import get_resnet18_backbone, get_resnet50_backbone, get_resnet34_backbone
from ml.visualization import save_sample, create_sample


def train_epoch(model, data_loader, loss_fn, optimizer, device, wandb_run):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0

    for batch_idx, (images, heatmaps) in enumerate(data_loader):
        images = images.to(device)
        heatmaps = heatmaps.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Compute loss
        loss = loss_fn(outputs, heatmaps)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


        if batch_idx % 10 == 0:
            wandb_run.log({"batch_loss": loss.item()})
            print(f'Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}')

    return running_loss / len(data_loader)


def validate(model, data_loader, loss_fn, device, epoch=0, save_vis=False, vis_dir="", wandb_run: wandb.Run = None):
    """Validate the model and optionally save visualizations"""
    model.eval()
    running_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch_idx, (images, heatmaps) in enumerate(data_loader):
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, heatmaps)

            running_loss += loss.item()
            batch_count += 1

            # Save 1 image per epoch for visualizing
            if save_vis and batch_idx % 100 == 0:
                filename = f'epoch_{epoch:03d}_idx_{batch_idx}.jpg'
                filepath = os.path.join(vis_dir, filename)

                result = create_sample(images[0].cpu(), outputs[0].cpu(), heatmaps[0].cpu())
                save_sample(result, filepath)

                wandb_image = wandb.Image(filepath, caption=filename)
                wandb_run.log({"visualizations": wandb_image})

    return running_loss / batch_count

def collate_fn(batch):
    """Filter out None batches to skip images without keypoints"""
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None
    return torch.stack([item[0] for item in batch]), torch.stack([item[1] for item in batch])


def main(model):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create datasets
    print('Loading datasets...')
    train_dataset = COCOKeypointDataset(
        DATA_PATH,
        split='train',
        img_size=IMG_SIZE,
        heatmap_size=HEATMAP_SIZE,
        max_samples=MAX_SAMPLES
    )

    val_dataset = COCOKeypointDataset(
        DATA_PATH,
        split='validation',
        img_size=IMG_SIZE,
        heatmap_size=HEATMAP_SIZE,
        max_samples=MAX_SAMPLES
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    # Initialize model
    print('Initializing model...')
    model = model.to(device)

    ckpt_dir = os.path.join(CHECKPOINT_DIR, model.name)
    vis_dir = os.path.join(VIS_DIR, model.name)
    # Create checkpoint directory
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    # Loss and optimizer
    loss_fn = KeypointMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA)

    wandb.login()

    # Project that the run is recorded to
    project = "dat255-pose-estimation"

    # Dictionary with hyperparameters
    config = {
        "model": model.name,
        'epochs': NUM_EPOCHS,
        'lr': LEARNING_RATE,
        "lr_scheduler_gamma": LR_SCHEDULER_GAMMA,
        "lr_scheduler_step_size": LR_SCHEDULER_STEP_SIZE,
        "batch_size": BATCH_SIZE,
        "image_size": IMG_SIZE,
        "num_keypoints": NUM_KEYPOINTS,
        "heatmap_size": HEATMAP_SIZE,
        "num_workers": NUM_WORKERS,
        "num_batches": len(train_loader),
        "num_val_batches": len(val_loader),
        "device": device,
        "dataset": "coco2017",
        "loss_fn": "MSELoss",
        "optimizer": "Adam",
    }

    with wandb.init(project=project, config=config, name=model.name) as wandb_run:

        # Training loop
        print('Starting training...')
        best_val_loss = float('inf')

        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}')
            wandb_run.log({"epoch": epoch + 1})

            # Train
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, wandb_run=wandb_run)

            # Validate and visualize every VIS_INTERVAL epochs
            should_visualize = (epoch + 1) % VIS_INTERVAL == 0
            val_loss = validate(model, val_loader, loss_fn, device, epoch=epoch, save_vis=should_visualize, vis_dir=vis_dir, wandb_run=wandb_run)

            # Update learning rate
            scheduler.step()

            end_time = time.time()
            epoch_time = end_time - start_time

            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Epoch Time: {epoch_time:.2f}s')
            wandb_run.log({"train_loss": train_loss, "val_loss": val_loss, "epoch_time": epoch_time})

            checkpoint_name = f'{model.name}_epoch_{epoch}.pth'
            checkpoint_path = os.path.join(ckpt_dir, checkpoint_name)

            torch.save(model.state_dict(), checkpoint_path)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(ckpt_dir, MODEL_SAVE_NAME)
                torch.save(model.state_dict(), checkpoint_path)

                print(f'Saved best model to {checkpoint_path}!')

                wandb_run.log_model(path=checkpoint_path, name=checkpoint_name)

        print('Training complete!')

if __name__ == '__main__':


    main(get_pose_convnet())

    main(get_pose_resnet101(pretrained=True))
    main(get_pose_resnet101(pretrained=False))

    main(get_pose_resnet152(pretrained=True))
    main(get_pose_resnet152(pretrained=False))

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights_backbone=ResNet50_Weights.IMAGENET1K_V1)
    model.name = "keypointrcnn_resnet50_fpn_pretrained"
    main(model)
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights_backbone=None, trainable_backbone_layers=5)
    model.name = "keypointrcnn_resnet50_fpn"
    main(model)
