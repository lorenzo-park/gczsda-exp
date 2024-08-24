from torch.autograd import Function
from torch.optim.lr_scheduler import _LRScheduler

import timm

import torch.nn as nn

from module.lenet import LeNet
from module.etc import Flatten
from module.mobilenet_v2 import mobilenet_v2


class DANNModel(nn.Module):
    def __init__(self, config, classes):
        super(DANNModel, self).__init__()
        self.backbone_name = config.backbone_name

        if self.backbone_name == "lenet":
            self.feature_extractor = LeNet()
            self.global_pool = nn.Identity()

            in_features = self.feature_extractor.num_features
            self.head = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=100),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=100, out_features=100),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=100, out_features=10),
            )
            self.discriminator = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=100),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=100, out_features=2),
            )
        elif self.backbone_name == "mobilenet_v2":
            self.feature_extractor, self.head = mobilenet_v2(num_classes=classes)
            self.global_pool = nn.Identity()
            self.hidden_dim = config.hidden_dim
            self.discriminator = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(self.feature_extractor.last_channel, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(self.hidden_dim, 2)
            )
        else:
            self.feature_extractor = timm.create_model(
                self.backbone_name,
                pretrained=True,
                num_classes=0,
                global_pool='',
            )
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.hidden_dim = config.hidden_dim

            in_features = self.feature_extractor.num_features
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, classes),
            )

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
        x = self.feature_extractor(x)

        if 'vit' not in self.backbone_name and \
            'swin' not in self.backbone_name and \
                'cait' not in self.backbone_name and \
                    'lenet' != self.backbone_name and \
                        'mobilenet_v2' != self.backbone_name:
            x = self.global_pool(x)
            x = x[:, :, 0, 0]
        outputs = self.head(x)

        return outputs

    def forward_train(self, x_src, x_tgt, lambda_p):
        x_src = self.feature_extractor(x_src)

        if 'vit' not in self.backbone_name and \
            'swin' not in self.backbone_name and \
                'cait' not in self.backbone_name and \
                    'lenet' != self.backbone_name and \
                        'mobilenet_v2' != self.backbone_name:
            x_src = self.global_pool(x_src)
            x_src = x_src[:, :, 0, 0]

        outputs_src = self.head(x_src)
        x_src_rev = GRL.apply(x_src, lambda_p)
        outputs_dsc_src = self.discriminator(x_src_rev)

        x_tgt = self.feature_extractor(x_tgt)

        if 'vit' not in self.backbone_name and \
            'swin' not in self.backbone_name and \
                'cait' not in self.backbone_name and \
                    'lenet' != self.backbone_name and \
                        'mobilenet_v2' != self.backbone_name:
            x_tgt = self.global_pool(x_tgt)
            x_tgt = x_tgt[:, :, 0, 0]

        x_tgt_rev = GRL.apply(x_tgt, lambda_p)
        outputs_dsc_tgt = self.discriminator(x_tgt_rev)

        return outputs_src, outputs_dsc_src, outputs_dsc_tgt

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


class GRL(Function):
  @staticmethod
  def forward(ctx, x, alpha):
    ctx.alpha = alpha

    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    output = grad_output.neg() * ctx.alpha

    return output, None


class DANNLRScheduler(_LRScheduler):
  def __init__(self, optimizer, init_lr, alpha, beta, total_steps):
    self.init_lr = init_lr
    self.alpha = alpha
    self.beta = beta
    self.total_steps = total_steps

    super(DANNLRScheduler, self).__init__(optimizer)

  def get_lr(self):
    current_iterations = self.optimizer._step_count
    p = float(current_iterations / self.total_steps)

    return [
        param["weight_lr"] * self.init_lr / (1 + self.alpha * p) ** self.beta
        for param in self.optimizer.param_groups
    ]