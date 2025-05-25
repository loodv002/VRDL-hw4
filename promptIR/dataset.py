from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random

class TrainDataset(Dataset):
    def __init__(self, root_dir, transform, crop=True, augmentation=True):
        super(TrainDataset, self).__init__()

        self.root = Path(root_dir)
        self.degraded_data_path = [
            path
            for path in (self.root / 'degraded').iterdir()
            if not path.stem.startswith('.')
        ]
        self.transform = transform

        self.crop = crop
        self.augmentation = augmentation
        self.W = 128

    @staticmethod
    def degraded_path_to_clean(degraded_path: Path):
        degrade_type, id = degraded_path.stem.split('.')[0].split('-')
        clean_name = f'{degrade_type}_clean-{id}.png'
        return degraded_path.parent.parent / 'clean' / clean_name
    
    def _random_crop(self, degraded_image: np.ndarray, clean_image: np.ndarray):
        h, w = degraded_image.shape[:2]
        i = random.randint(0, h - self.W)
        j = random.randint(0, w - self.W)

        degraded_image = degraded_image[i:i+self.W, j:j+self.W, :]
        clean_image = clean_image[i:i+self.W, j:j+self.W, :]
        return degraded_image, clean_image
    
    def _random_flip(self, degraded_image: np.ndarray, clean_image: np.ndarray):
        if random.randint(0, 1):
            degraded_image = degraded_image[:, ::-1, :]
            clean_image = clean_image[:, ::-1, :]
        return degraded_image, clean_image
    
    def _random_rotate(self, degraded_image: np.ndarray, clean_image: np.ndarray):
        k = random.randint(0, 3)
        degraded_image = np.rot90(degraded_image, k)
        clean_image = np.rot90(clean_image, k)
        return degraded_image, clean_image

    def __getitem__(self, i):
        degraded_path = self.degraded_data_path[i]
        clean_path = self.degraded_path_to_clean(degraded_path)

        degraded_image = np.array(Image.open(degraded_path).convert('RGB'))
        clean_image = np.array(Image.open(clean_path).convert('RGB'))

        if self.augmentation:
            degraded_image, clean_image = self._random_flip(degraded_image, clean_image)
            degraded_image, clean_image = self._random_rotate(degraded_image, clean_image)

        if self.crop:
            degraded_image, clean_image = self._random_crop(degraded_image, clean_image)

        degraded_image = self.transform(degraded_image.copy())
        clean_image = self.transform(clean_image.copy())


        return degraded_image, clean_image

    def __len__(self):
        return len(self.degraded_data_path)
    
class TestDataset(Dataset):
    def __init__(self, root_dir, transform):
        super(TestDataset, self).__init__()

        self.root = Path(root_dir)
        self.degraded_data_path = [
            path
            for path in (self.root / 'degraded').iterdir()
            if not path.stem.startswith('.')
        ]
        self.transform = transform

    def __getitem__(self, i):
        degraded_path = self.degraded_data_path[i]
        degraded_image = Image.open(degraded_path).convert('RGB')
        degraded_image = self.transform(degraded_image)

        return degraded_path.name, degraded_image

    def __len__(self):
        return len(self.degraded_data_path)