import timm
import copy

import torch.nn as nn

from pl_module.timm import LitTimm
from module.etc import Flatten


class ADDAModel(nn.Module):
    def __init__(self, config):
        super(ADDAModel, self).__init__()
        self.backbone_name = config.backbone_name

        assert config.pretrained is not None
        model = LitTimm.load_from_checkpoint(config.pretrained, config=config).model

        self.feature_extractor_src = model.feature_extractor
        self.feature_extractor_tgt = copy.deepcopy(model.feature_extractor)

        self.head = model.head

        if self.backbone_name == "lenet":
            in_features = self.feature_extractor_src.num_features
            self.hidden_dim = 100
            self.global_pool = nn.Identity()
            self.discriminator = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=100),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=100, out_features=2),
            )
        elif self.backbone_name == "mobilenet_v2":
            in_features = self.feature_extractor_src.last_channel
            self.global_pool = nn.Identity()
            self.hidden_dim = config.hidden_dim
            self.discriminator = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(in_features, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(self.hidden_dim, 2)
            )
        else:
            in_features = self.feature_extractor_src.num_features
            self.hidden_dim = config.hidden_dim
            self.global_pool = nn.AdaptiveAvgPool2d(1)

            self.discriminator = nn.Sequential(
                nn.Linear(in_features, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(self.hidden_dim, 2)
            )

    def forward(self, x):
        features = self.feature_extractor_tgt(x)

        if 'vit' not in self.backbone_name and \
            'swin' not in self.backbone_name and \
                'cait' not in self.backbone_name and \
                    'lenet' != self.backbone_name and \
                        'mobilenet_v2' != self.backbone_name:
            features = self.global_pool(features)
            features = features[:, :, 0, 0]

        x = self.head(features)

        return x

    def forward_gan(self, x_tgt):
        x_tgt = self.feature_extractor_tgt(x_tgt)

        if 'vit' not in self.backbone_name and \
            'swin' not in self.backbone_name and \
                'cait' not in self.backbone_name and \
                    'lenet' != self.backbone_name and \
                        'mobilenet_v2' != self.backbone_name:
            x_tgt = self.global_pool(x_tgt)
            x_tgt = x_tgt[:, :, 0, 0]

        outputs_dsc_tgt = self.discriminator(x_tgt)

        return outputs_dsc_tgt

    def forward_dsc(self, x_src, x_tgt):
        x_src = self.feature_extractor_src(x_src)
        x_tgt = self.feature_extractor_tgt(x_tgt)

        if 'vit' not in self.backbone_name and \
            'swin' not in self.backbone_name and \
                'cait' not in self.backbone_name and \
                    'lenet' != self.backbone_name and \
                        'mobilenet_v2' != self.backbone_name:
            x_src = self.global_pool(x_src)
            x_src = x_src[:, :, 0, 0]
            x_tgt = self.global_pool(x_tgt)
            x_tgt = x_tgt[:, :, 0, 0]

        outputs_dsc_src = self.discriminator(x_src)
        outputs_dsc_tgt = self.discriminator(x_tgt)

        return outputs_dsc_src, outputs_dsc_tgt
