import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio

from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, Union

from .model import WrappedPromptIR
from .loss_utils import WeightedLoss

class Trainer:
    def __init__(self):
        self.device = torch.device('cuda' 
                                   if torch.cuda.is_available()
                                   else 'cpu')
        
    def _train_epoch(self, model, optimizer, loss_function, data_loader) -> float:
        train_loss = 0

        model.train()
        for degraded_images, clean_images in tqdm(data_loader, desc='  Training phase'):
            degraded_images = degraded_images.to(self.device)
            clean_images = clean_images.to(self.device)
            restored_images = model(degraded_images)

            loss = loss_function(clean_images, restored_images)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        train_loss /= len(data_loader)
        return train_loss

    @torch.no_grad()
    def _val_epoch(self, model, loss_function, data_loader) -> Tuple[float, float]:
        val_loss = 0
        psnr = PeakSignalNoiseRatio(data_range=1.0)
        
        model.eval()
        for degraded_images, clean_images in tqdm(data_loader, desc='  Validation phase'):
            degraded_images = degraded_images.to(self.device)
            clean_images = clean_images.to(self.device)
            restored_images = model(degraded_images)

            loss = loss_function(clean_images, restored_images)
            psnr.update(clean_images.cpu(), torch.clamp(restored_images.cpu(), 0, 1))

            val_loss += loss.detach().item()

        val_loss /= len(data_loader)
        val_psnr = psnr.compute().item()

        return val_psnr, val_loss

    def train(self,
              model: WrappedPromptIR,
              train_loader: DataLoader,
              val_loader: DataLoader,
              checkpoint_dir: Union[Path, str],
              max_epoches: int,
              optimizer: optim.Optimizer,
              scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
              early_stop: bool = True):
        
        checkpoint_dir = Path(checkpoint_dir)

        print(f'Train model by {self.device}')
        
        loss_function = WeightedLoss(self.device)

        model = model.to(self.device)

        min_val_loss = float('inf')
        val_loss_increase_count = 0
        
        train_losses = []
        val_losses = []
        val_PSNRs = []

        for epoch in range(max_epoches):
            print(f'Epoch {epoch}:')
            print(f'  Learning rate: {optimizer.param_groups[0]["lr"]}')

            train_loss = self._train_epoch(
                model,
                optimizer,
                loss_function,
                train_loader,
            )

            val_PSNR, val_loss = self._val_epoch(
                model,
                loss_function,
                val_loader,
            )

            print(f'  Training loss: {train_loss:.5f}')
            print(f'  Validation loss: {val_loss:.5f}')
            print(f'  Validation PSNR: {val_PSNR:.5f}')

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_PSNRs.append(val_PSNR)

            if scheduler: scheduler.step()

            model_path = checkpoint_dir / f'{model.model_name}_epoch_{epoch}.pth'
            torch.save(model.state_dict(), model_path)

            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                val_loss_increase_count = 0
            else:
                val_loss_increase_count += 1

            if val_loss_increase_count >= 2 and early_stop:
                print('Loss increased, training stopped.')
                break

        else:
            print('Max epoches reached.')

        print(f'Train losses: {train_losses}')
        print(f'Val losses: {val_losses}')
        print(f'Val PSNRs: {val_PSNRs}')