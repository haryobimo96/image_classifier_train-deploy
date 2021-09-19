import torch
import torchvision.models as models
from torch import nn

class VGG16NetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_features: bool = True,
        **kwargs
        ):
        super().__init__()
        self.base_model = models.vgg16(pretrained = pretrained)

        in_features = self.base_model.classifier[6].in_features

        if freeze_features:
            frozen_layers = [self.base_model.features]
            for layer in frozen_layers:
                for name, value in layer.named_parameters():
                    value.requires_grad = False
          
        self.base_model.classifier[6].in_features = torch.nn.Linear(
            in_features = in_features, out_features = num_classes,
            bias = True,
        )
          
    def forward(self, x):
        logits = self.base_model(x)
        return logits

          
