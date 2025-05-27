import torch
import numpy as np

import argparse
import os
import yaml
from tqdm import tqdm
import zipfile
from pathlib import Path

from promptIR import WrappedPromptIR
from promptIR.transform import val_transform, tensor_to_image
from promptIR.dataset import TestDataset

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config.yml', help='config file path')
parser.add_argument('--checkpoint', required=True, help='checkpoint name')
args = parser.parse_args()

config_path = args.config
checkpoint = args.checkpoint

print(f'Config file: {config_path}')
print(f'Checkpoint: {checkpoint}')

if not os.path.exists(config_path):
    print(f'Config file "{config_path}" not exist.')
    exit()

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = Path(config['path']['DATA_DIR'])
MODEL_DIR = Path(config['path']['MODEL_DIR'])
OUTPUT_DIR = Path(config['path']['OUTPUT_DIR'])

BATCH_SIZE = config['train']['BATCH_SIZE']

test_dataset = TestDataset(DATA_DIR / 'test', val_transform)

device = torch.device('cuda' 
                      if torch.cuda.is_available()
                      else 'cpu')

model = WrappedPromptIR()
model.load_state_dict(torch.load(MODEL_DIR / f'{checkpoint}.pth'))
model.to(device)
model.eval()

images = {}
output_npz_path = OUTPUT_DIR / f'{checkpoint}_ens8.npz'
output_zip_path = OUTPUT_DIR / f'{checkpoint}_ens8.zip'

model.to(device)
model.eval()
with torch.no_grad():
    for image_name, degraded_image in tqdm(test_dataset): # type: ignore
        degraded_image = degraded_image.to(device).unsqueeze(0)

        restored_images = []

        for direction in range(4):
            rotated_image = torch.rot90(degraded_image, direction, (2, 3))
            restored_image = model(rotated_image)
            restored_image = torch.rot90(restored_image, -direction, (2, 3))
            restored_images.append(restored_image.squeeze(0))
        for direction in range(4):
            rotated_image = torch.rot90(torch.transpose(degraded_image, 2, 3), direction, (2, 3))
            restored_image = model(rotated_image)
            restored_image = torch.transpose(torch.rot90(restored_image, -direction, (2, 3)), 2, 3)
            restored_images.append(restored_image.squeeze(0))

        restored_image = torch.stack(restored_images).mean(dim=0)
        restored_image = tensor_to_image(restored_image)
        images[image_name] = restored_image

np.savez(output_npz_path, **images)

with zipfile.ZipFile(output_zip_path, mode='w') as f:
    f.write(output_npz_path, 'pred.npz')