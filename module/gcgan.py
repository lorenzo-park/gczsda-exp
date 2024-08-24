import torch
import torch.nn as nn
import torch.nn.functional as F

from module.etc import ImagePool
from pl_module.timm import LitTimm


class GcGANModel(nn.Module):
    def __init__(self, config):
        super(GcGANModel, self).__init__()

        self.semantic_consistency = config.lambda_sem > 0
        self.identity_loss = config.lambda_idt > 0
        self.identity_semantic_consistency = config.lambda_sem_idt > 0
        self.fix_block_up = config.fix_block_up
        self.img_size = config.img_size
        self.transformation = self.get_transformation(config.transformation)

        self.generator_ts = G(
            num_blocks=config.num_blocks,
            channels=config.channels
        )
        self.discriminator = D(
            hidden_dim=config.hidden_dim_dsc,
            channels=config.channels,
        )
        self.discriminator_gc = D(
            hidden_dim=config.hidden_dim_dsc,
            channels=config.channels,
        )

        # Image history replay
        self.image_pool = ImagePool(config.batch_size * config.grad_accum // 2)
        self.image_pool_gc = ImagePool(config.batch_size * config.grad_accum // 2)

        assert config.pretrained is not None
        self.pretrained = LitTimm.load_from_checkpoint(config.pretrained, config=config).model
        self.pretrained.eval()

    def forward(self, inputs_src):
        return self.pretrained(inputs_src)

    def forward_gan(self, inputs_src, inputs_tgt):
        inputs_tgt_gc = self.transformation(inputs_tgt, 0)

        if self.identity_loss:
            inputs_src_self_translated = self.generator_ts(inputs_src)
        else:
            inputs_src_self_translated = None

        if self.identity_semantic_consistency:
            inputs_src_self_translated_sem_idt = self.generator_ts(inputs_src, fix_block_up=self.fix_block_up)
            outputs_src_self_translated = self.pretrained(inputs_src_self_translated_sem_idt)
            outputs_src = self.pretrained(inputs_src)
        else:
            outputs_src_self_translated, outputs_src = None, None

        inputs_src_translated = self.generator_ts(inputs_tgt)
        outputs_dsc_src_translated = self.discriminator(inputs_src_translated)

        inputs_src_gc_translated = self.generator_ts(inputs_tgt_gc)
        outputs_dsc_src_gc_translated = self.discriminator_gc(inputs_src_gc_translated)

        return inputs_src_translated, self.transformation(inputs_src_gc_translated, 1), \
            inputs_src_gc_translated, self.transformation(inputs_src_translated, 0), \
                outputs_dsc_src_translated, outputs_dsc_src_gc_translated, \
                    inputs_src_self_translated, outputs_src_self_translated, outputs_src

    def forward_dsc(self, inputs_src, inputs_tgt, p=1):
        inputs_tgt_gc = self.transformation(inputs_tgt, 0)

        inputs_src_translated = self.generator_ts(inputs_tgt)
        inputs_src_gc_translated = self.generator_ts(inputs_tgt_gc)

        inputs_src_translated = (1 - p) * inputs_tgt + p * inputs_src_translated
        inputs_src_gc_translated = (1 - p) * inputs_tgt_gc + p * inputs_src_gc_translated

        # Train discriminator
        inputs_src_gc = self.transformation(inputs_src, 0)
        outputs_dsc_src = self.discriminator(inputs_src)
        outputs_dsc_src_translated = self.discriminator(
            self.image_pool.query(inputs_src_translated)
        )

        outputs_dsc_src_gc = self.discriminator_gc(inputs_src_gc)
        outputs_dsc_src_gc_translated = self.discriminator_gc(
            self.image_pool_gc.query(inputs_src_gc_translated)
        )

        return outputs_dsc_src, outputs_dsc_src_translated, \
            outputs_dsc_src_gc, outputs_dsc_src_gc_translated

    def translate(self, x, random_noise=None):
        return self.generator_ts(x, random_noise=random_noise)

    def get_feature(self, x):
        feature = self.generator_ts.get_feature(x)[:,:,0,0]
        return feature

    def get_transformation(self, transformation):
        if transformation == "rotate":
            return self.rot90
        elif transformation == "flip":
            return self.vflip

    def rot90(self, tensor, direction):
        # tensor = tensor.transpose(2, 3)
        # size = self.img_size
        # inv_idx = torch.arange(size-1, -1, -1).long()
        # inv_idx = inv_idx.to(tensor.device)
        # if direction == 0:
        #     tensor = torch.index_select(tensor, 3, inv_idx)
        # else:
        #     tensor = torch.index_select(tensor, 2, inv_idx)
        # return tensor
        if direction == 0:
            return torch.rot90(tensor, -1, [-2, -1])
        else:
            return torch.rot90(tensor, 1, [-2, -1])

    def vflip(self, tensor, _):
        return torch.fliplr(tensor)

    def get_parameters_generator(self):
        return list(self.generator_ts.parameters())

    def get_parameters_discriminator(self):
        return list(self.discriminator.parameters()) + \
            list(self.discriminator_gc.parameters())

    def get_parameters_pretrained(self):
        return list(self.pretrained.parameters())


class Block(nn.Module):
    def __init__(self, channels=256, norm=nn.InstanceNorm2d):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            norm(channels),
            nn.ReLU(inplace=True),
#             DebugLayer(),

            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            norm(channels),
#             DebugLayer(),
        )

    def forward(self, x):
        out = x + self.block(x)
        return out


class G(nn.Module):
    """
    CREDIT: https://arxiv.org/pdf/1711.03213.pdf (CyCADA)
    Figure 7 in Section 6.2
    """
    def __init__(self, num_blocks, channels):
        super(G, self).__init__()

        self.block_down = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, 64, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
#             DebugLayer(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
#             DebugLayer(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
#             DebugLayer(),
        )


        blocks = []
        for _ in range(num_blocks):
            blocks.append(Block())
        self.blocks = nn.Sequential(*blocks)

        self.block_up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, inputs, fix_block_up=False, random_noise=None):
        if fix_block_up:
            with torch.no_grad():
                x = self.block_down(inputs)
                x = self.blocks(x)
        else:
            x = self.block_down(inputs)
            x = self.blocks(x)
        if random_noise is not None:
            # import matplotlib.pyplot as plt
            # plt.imshow(x[10][0].clone().detach().numpy())
            # plt.show()
            # print(x.shape)
            # print(x[10][0])
            noise = torch.normal(torch.mean(x, dim=0), torch.std(x, dim=0))
            x += noise * random_noise
        x = self.block_up(x)

        return x

    def get_feature(self, x):
        x = self.block_down(x)
        x = self.blocks(x)

        return F.avg_pool2d(x, x.size()[2:])


class D(nn.Module):
    def __init__(self, hidden_dim, channels):
        super(D, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # DebugLayer(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # DebugLayer(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # DebugLayer(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # DebugLayer(),
            nn.Conv2d(hidden_dim * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, x.size()[2:])
        return x.squeeze()
class DebugLayer(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x
