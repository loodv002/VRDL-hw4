import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

# https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49?permalink_comment_id=4001178
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(vgg19(weights=VGG19_Weights.DEFAULT).features[:4].eval()) # type: ignore
        blocks.append(vgg19(weights=VGG19_Weights.DEFAULT).features[4:9].eval()) # type: ignore
        blocks.append(vgg19(weights=VGG19_Weights.DEFAULT).features[9:16].eval()) # type: ignore
        blocks.append(vgg19(weights=VGG19_Weights.DEFAULT).features[16:23].eval()) # type: ignore
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class WeightedLoss:
    def __init__(self, 
                 device: torch.device,
                 l1_weight: float = 10.0,
                 ssim_weight: float = 1.0,
                 perceptual_weight: float = 1.0):
        
        self.l1 = nn.L1Loss().to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.perceptual = VGGPerceptualLoss(resize=False).to(device)

        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight

    def __call__(self, preds: torch.Tensor, gts: torch.Tensor):
        return (
            self.l1_weight * (self.l1(preds, gts)) +
            self.ssim_weight * (1 - self.ssim(preds, gts)) +
            self.perceptual_weight * (self.perceptual(preds, gts))
        )