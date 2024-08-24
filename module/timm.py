import timm

import torch.nn as nn

from module.lenet import LeNet
from module.mobilenet_v2 import mobilenet_v2


class TimmModel(nn.Module):
    def __init__(self, config, classes):
        super(TimmModel, self).__init__()
        self.backbone_name = config.backbone_name

        if self.backbone_name == "lenet":
            self.feature_extractor = LeNet()
            self.global_pool = nn.Identity()

            in_features = self.feature_extractor.num_features
            self.head = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=100),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=100, out_features=config.output_num),
            )
        elif self.backbone_name == "mobilenet_v2":
            self.feature_extractor, self.head = mobilenet_v2(num_classes=classes)
            self.global_pool = nn.Identity()
        else:
            self.feature_extractor = timm.create_model(
                self.backbone_name,
                pretrained=True,
                num_classes=0,
                global_pool='',
                in_chans=config.channels,
            )

            self.global_pool = nn.AdaptiveAvgPool2d(1)

            self.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.feature_extractor.num_features, classes),
            )


    def forward(self, x):
        x = self.feature_extractor(x)

        if 'vit' not in self.backbone_name and \
            'swin' not in self.backbone_name and \
                'cait' not in self.backbone_name and \
                    'lenet' != self.backbone_name and \
                        'mobilenet_v2' != self.backbone_name:
            x = self.global_pool(x)
            x = x[:, :, 0, 0]

        x = self.head(x)

        return x

    def get_feature(self, x):
        x = self.feature_extractor(x)

        if 'vit' not in self.backbone_name and \
            'swin' not in self.backbone_name and \
                'cait' not in self.backbone_name and \
                    'lenet' != self.backbone_name and \
                        'mobilenet_v2' != self.backbone_name:
            x = self.global_pool(x)
            x = x[:, :, 0, 0]

        return x


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
