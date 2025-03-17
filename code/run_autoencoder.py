# EXAMPLE USAGE:
# python run_autoencoder.py configs/default.yaml

import numpy as np
import sys
import os
import yaml
import gc
import torch
import lightning as L

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from autoencoder import Autoencoder
from patchdataset import PatchDataset
from data import make_data

# ------------------------ Load Config ------------------------
print("Loading config file...")
config_path = sys.argv[1]
assert os.path.exists(config_path), f"Config file not found: {config_path}"
config = yaml.safe_load(open(config_path, "r"))

# Optional: Set matmul precision (recommended by PyTorch for GPUs with Tensor Cores)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

# Clean up memory before start
gc.collect()
torch.cuda.empty_cache()

# ------------------------ Prepare Data ------------------------
print("Making patch data...")
_, patches = make_data(patch_size=config["data"]["patch_size"])
all_patches = [patch for image_patches in patches for patch in image_patches]

print(f"Total patches loaded: {len(all_patches)}")

# Random train/val split
train_bool = np.random.rand(len(all_patches)) < 0.8
train_idx = np.where(train_bool)[0]
val_idx = np.where(~train_bool)[0]

train_patches = [all_patches[i] for i in train_idx]
val_patches = [all_patches[i] for i in val_idx]

train_dataset = PatchDataset(train_patches)
val_dataset = PatchDataset(val_patches)

dataloader_train = DataLoader(train_dataset, **config["dataloader_train"])
dataloader_val = DataLoader(val_dataset, **config["dataloader_val"])

# ------------------------ Model ------------------------
print("Initializing model...")
model = Autoencoder(
    optimizer_config=config["optimizer"],
    patch_size=config["data"]["patch_size"],
    **config["autoencoder"],
)
print(model)

# ------------------------ Callbacks ------------------------
print("Preparing callbacks...")
checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

callbacks = [checkpoint_callback]

# Add EarlyStopping if specified
if "early_stopping" in config:
    early_stop_callback = EarlyStopping(**config["early_stopping"])
    callbacks.append(early_stop_callback)

# ------------------------ Logger ------------------------
print("Setting up WandB logger...")
wandb_logger = WandbLogger(config=config, **config["wandb"])

# ------------------------ Trainer ------------------------
print("Starting training...")
trainer = L.Trainer(
    logger=wandb_logger,
    callbacks=callbacks,
    **config["trainer"]
)

trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

# Clean up at end
gc.collect()
torch.cuda.empty_cache()
