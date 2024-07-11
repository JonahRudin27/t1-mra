import argparse
import os
import logging
import multiprocessing
import psutil

from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter

from T1mra_dataset import T1w2MraDataset_scans
from PerceptualLoss_3d import PerceptualLoss_3d, VGG16FeatureExtractor
from UNet import UNet
from train_utils import train_scans, validate, tensorboard_write, RandomRotation90

if __name__ == "__main__":

    writer = SummaryWriter()

    # Get training args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dir "
                        "containing training data. "
                        "Must have train, valid, test subdirectories")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch"
                        "size for training")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number "
                        "of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Number "
                        "of epochs to wait for improvement before stopping")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="Minimum change to qualify as an improvement")
    parser.add_argument("--num_workers", type=int, default=-1, help="Number "
                        "of workers for dataloader")
    parser.add_argument("--preload_dtype", type=str, default="float32",)
    parser.add_argument("--early_stopping", type=bool, default=False,
                        help="Use early stopping")
    parser.add_argument("--force_single_gpus", type=bool, default=False,
                        help="Force single GPU usage")
    args = parser.parse_args()

    # Early stopping parameters
    if args.early_stopping:
        patience = args.patience
        min_delta = args.min_delta
        best_val_loss = float('inf')
        epochs_no_improve = 0

    # logging
    logging.basicConfig(filename='training.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # setup perceptual loss
    feature_extractor = VGG16FeatureExtractor()
    feature_extractor.to(device)
    perceptual_loss = PerceptualLoss_3d(feature_extractor, MSELoss)

    # def transforms
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        RandomRotation90(),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])
    # check cpu count
    if args.num_workers == -1:
        num_workers = multiprocessing.cpu_count() - 1
    else:
        num_workers = args.num_workers

    # def datasets/dataloaders
    print(f'Loading datasets from {args.data_dir}')
    train_dataset = T1w2MraDataset_scans(os.path.join(args.data_dir, "train", "T1W"),
                                   os.path.join(args.data_dir, "train", "MRA"),
                                   transform=train_transform,
                                   preload_dtype=args.preload_dtype,
                                   slice_width=1)
    valid_dataset = T1w2MraDataset_scans(os.path.join(args.data_dir, "valid", "T1W"),
                                   os.path.join(args.data_dir, "valid", "MRA"),
                                   transform=train_transform,
                                   preload_dtype=args.preload_dtype,
                                   slice_width=1)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=False)

    # Print current memory usage
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss
    print(f"Current memory usage: {current_memory / (1024**3)} GB")

    # def model
    z_model = UNet(1, 1)
    y_model = UNet(1, 1)
    x_model = UNet(1, 1)

    models = [z_model, y_model, x_model]

    def config_model(model):
        if args.force_single_gpus:
            gpu_count = 1
        else:
            gpu_count = torch.cuda.device_count()
            print(f"Found {gpu_count} GPUs")
            if gpu_count > 1:
                model = torch.nn.DataParallel(model)
                print(f"Using DataParallel with {gpu_count} GPUs")

        model.to(device)

        # def optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=0.5, patience=10)
        num_epochs = args.num_epochs

        # load checkpoint if exists
        if os.path.exists("model_checkpoint.pth"):
            print("Model checkpoint found, loading")
            checkpoint = torch.load("model_checkpoint.pth")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            print(f'Loaded model, starting from epoch {start_epoch}')
        else:
            start_epoch = 0
        
        return optimizer, scheduler, start_epoch, num_epochs
    
    class TrainingSession:
        def __init__(self, optimizer, scheduler, start_epoch, num_epochs):
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.start_epoch = start_epoch
            self.num_epochs = num_epochs

    training = [TrainingSession(config_model(z_model)), 
                TrainingSession(config_model(y_model)), 
                TrainingSession(config_model(x_model))]

    print(f'Z model: Starting training for {training[0].num_epochs} epochs \n')
    print(f'y model: Starting training for {training[1].num_epochs} epochs \n')
    print(f'x model: Starting training for {training[2].num_epochs} epochs \n')
    # training loop
    for epoch in range(training[0].start_epoch, training[0].num_epochs):
        train_loss = [0.0, 0.0, 0.0]
        val_loss = [0.0, 0.0, 0.0]
        for i in range(3):
            train_loss[i] = train_scans(models[i], train_dataloader, PerceptualLoss_3d.get_loss,
                            training[i].optimizer, device)
            val_loss[i] = validate(models[i], valid_dataloader,
                                PerceptualLoss_3d.get_loss, device)

            training[i].scheduler.step(val_loss[i])

            print(f"Model {i}: Epoch {epoch+1}, Loss: {train_loss}, Val Loss: {val_loss}")
            logging.info(f"Model {i}: Epoch {epoch+1}, Loss: {train_loss}, "
                        f"Val Loss: {val_loss} "
                        f"LR: {training[i].scheduler.get_last_lr()}")

            tensorboard_write(writer, device, train_loss, val_loss, epoch+1,
                            models[i], valid_dataloader,
                            num_images=args.batch_size,
                            adam_optim=training[i].optimizer)

            # save model checkpoint
            if args.early_stopping:
                if best_val_loss - val_loss > min_delta:
                    best_val_loss = val_loss
                    epochs_no_improve = 0

                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': models[i].state_dict(),
                        'optimizer_state_dict': training[i].optimizer.state_dict()
                    }, "model_checkpoint.pth")

                else:
                    epochs_no_improve += 1

                if epochs_no_improve == patience:
                    print("Early stopping")
                    logging.info(f'Early stopping at epoch {epoch+1}')
                    break
            else:
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': models[i].state_dict(),
                    'optimizer_state_dict': training[i].optimizer.state_dict()
                }, "model_checkpoint.pth")
