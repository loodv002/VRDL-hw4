import torch
import torchvision.transforms as transforms
import numpy as np

train_transform = transforms.Compose([
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToTensor()
])

def tensor_to_image(image_tensor: torch.Tensor):
    image_np = torch.clamp(image_tensor, 0, 1).detach().cpu().numpy()
    image_np = np.round(image_np * 255).astype(np.uint8)
    return image_np