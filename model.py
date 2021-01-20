import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary


from vgg import VGG

pretrained_model = torch.load('pretrained models/vgg_voc_1e-05lr_0.04111loss_0.99534acc.pth')


class FeatureExtractedVGG(nn.Module):
    def __init__(self):
        super(FeatureExtractedVGG, self).__init__()
        self.conv_layers = pretrained_model.conv_layers
        self.avg_pool = pretrained_model.avg_pool
        self.classifier = pretrained_model.classifier
        self.box_reg = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4),
        )
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avg_pool(x)

        x = x.view(x.shape[0], -1)

        cls = self.classifier(x)
        cls = self.softmax(cls)

        bbox = self.box_reg(x)
        bbox = self.sigmoid(bbox)

        return cls, bbox
