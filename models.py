from torchvision.models import resnet18
from config import Config
import torch
from torch import nn
from torch import functional as F


def resnet(num_classes=32, pretrained=False):
    model = resnet18(pretrained=False)
    if pretrained is True:
        model.load_state_dict(torch.load(Config.resnet18_path))
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    return model


class ContrastNet(nn.Module):
    def __init__(self):
        super(ContrastNet, self).__init__()
        self.backbone = resnet(num_classes=32, pretrained=True)
        self.mlp = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8)
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.mlp(out)
        return out
