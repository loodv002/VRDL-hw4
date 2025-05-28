import os
import yaml
from pathlib import Path
import shutil
import random
import argparse

from typing import List

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config.yml', help='config file path')
args = parser.parse_args()

config_path = args.config

print(f'Config file: {config_path}')

if not os.path.exists(config_path):
    print(f'Config file "{config_path}" not exist.')
    exit()

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = Path(config['path']['DATA_DIR'])
N_VAL_DATA = config['data']['N_VAL_DATA'] // 2

original_train_dir = DATA_DIR / 'train'
splitted_train_dir = DATA_DIR / 'train_splitted'
splitted_val_dir = DATA_DIR / 'val_splitted'

if splitted_train_dir.exists():
    shutil.rmtree(splitted_train_dir)
if splitted_val_dir.exists():
    shutil.rmtree(splitted_val_dir)

os.makedirs(splitted_train_dir)
os.makedirs(splitted_train_dir / 'clean')
os.makedirs(splitted_train_dir / 'degraded')
os.makedirs(splitted_val_dir)
os.makedirs(splitted_val_dir / 'clean')
os.makedirs(splitted_val_dir / 'degraded')

def move_dataset(splitted_data: List[Path], dest_dir: Path):
    for clean_path in splitted_data:
        degrade_type, id = clean_path.stem.split('_clean-')
        degraded_path = clean_path.parent.parent / 'degraded' / f'{degrade_type}-{id}.png'

        shutil.copy(clean_path, dest_dir / 'clean')
        shutil.copy(degraded_path, dest_dir / 'degraded')


for degrade_type in ['rain', 'snow']:
    original_train_data = [
        path
        for path in (original_train_dir / 'clean').iterdir()
        if path.stem.startswith(degrade_type)
    ]
    splitted_val_data = random.sample(original_train_data, N_VAL_DATA)
    splitted_train_data = [
        path
        for path in original_train_data 
        if path not in splitted_val_data
    ]
    move_dataset(splitted_train_data, splitted_train_dir)
    move_dataset(splitted_val_data, splitted_val_dir)