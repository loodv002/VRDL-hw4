import torch
from torch.utils.data import DataLoader

import argparse
import yaml
import os
from pathlib import Path

from promptIR import WrappedPromptIR, Trainer
from promptIR.dataset import TrainDataset
from promptIR.transform import train_transform, val_transform

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config.yml', help='config file path')
args = parser.parse_args()

config_path = args.config
if not os.path.exists(config_path):
    print(f'Config file "{config_path}" not exist.')
    exit()

print(f'Use config file "{config_path}"')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = Path(config['path']['DATA_DIR'])
MODEL_DIR = Path(config['path']['MODEL_DIR'])
OUTPUT_DIR = Path(config['path']['OUTPUT_DIR'])

BATCH_SIZE = config['train']['BATCH_SIZE']
LEARNING_RATE = config['train']['LEARNING_RATE']
MAX_EPOCHES = config['train']['MAX_EPOCHES']
EARLY_STOP = config['train']['EARLY_STOP']

train_dataset = TrainDataset(DATA_DIR / 'train_splitted', train_transform)
val_dataset = TrainDataset(DATA_DIR / 'val_splitted', val_transform, crop=False)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=2)

model = WrappedPromptIR()

print(f'Model name: {model.model_name}')

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

trainer = Trainer()
trainer.train(
    model,
    train_dataloader,
    val_dataloader,
    MODEL_DIR,
    MAX_EPOCHES,
    optimizer,
    scheduler,
    EARLY_STOP,
)