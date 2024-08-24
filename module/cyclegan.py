import torch
import torch.nn as nn
import torch.nn.functional as F

from module.etc import ImagePool
from pl_module.timm import LitTimm
from module.timm import TimmModel


class CycleGANModel(nn.Module):
    def __init__(self, config):
        super(CycleGANModel, self).__init__()

        self.semantic_consistency = config.lambda_sem > 0
        self.identity_loss = config.lambda_idt > 0
        self.cycle_consistency_type = config.cycle_consistency_type
        self.use_progressive_mixup = config.use_progressive_mixup
        self.mixup_thres = config.mixup_thres
        self.eps = config.eps

        self.generator_st = G(
            num_blocks=config.num_blocks,
            channels=config.channels
        )
        self.generator_ts = G(
            num_blocks=config.num_blocks,
            channels=config.channels
        )

        self.discriminator_src = D(
            hidden_dim=config.hidden_dim_dsc,
            channels=config.channels,
        )
        self.discriminator_tgt = D(
            hidden_dim=config.hidden_dim_dsc,
            channels=config.channels,
        )

        # Image history replay
        self.image_pool_src = ImagePool(config.batch_size * config.grad_accum // 2)
        self.image_pool_tgt = ImagePool(config.batch_size * config.grad_accum // 2)

        
        if self.semantic_consistency:
            assert config.pretrained is not None
            self.pretrained = LitTimm.load_from_checkpoint(config.pretrained, config=config).model
            self.pretrained.eval()
        else:
            self.pretrained = TimmModel(config, classes=config.num_classes)
        

    def forward(self, inputs_tgt):
        return self.pretrained(inputs_tgt)

    def forward_gan(self, inputs_src, inputs_tgt):
        inputs_src_translated = self.generator_ts(inputs_tgt)
        inputs_tgt_translated = self.generator_st(inputs_src)

        if self.cycle_consistency_type == "geometric":
            inputs_src_cycled = self.generator_ts(inputs_tgt_translated)
            inputs_tgt_cycled = self.generator_st(inputs_src_translated)
        else:
            inputs_src_cycled = self.pretrained(self.generator_ts(inputs_tgt_translated))
            inputs_tgt_cycled = self.pretrained(self.generator_st(inputs_src_translated))

        # outputs_dsc_src = self.discriminator_src(inputs_src)
        outputs_dsc_src_translated = self.discriminator_src(inputs_src_translated)

        # outputs_dsc_tgt = self.discriminator_tgt(inputs_tgt)
        outputs_dsc_tgt_translated = self.discriminator_tgt(inputs_tgt_translated)

        if self.semantic_consistency:
            outputs_src_translated = self.pretrained(inputs_src_translated)
            outputs_tgt_translated = self.pretrained(inputs_tgt_translated)
            outputs_src = self.pretrained(inputs_src)
            outputs_tgt = self.pretrained(inputs_tgt)
        else:
            outputs_src_translated, outputs_tgt_translated, outputs_src, outputs_tgt = None, None, None, None

        if self.identity_loss:
            inputs_src_translated_self = self.generator_ts(inputs_src)
            inputs_tgt_translated_self = self.generator_st(inputs_tgt)
        else:
            inputs_src_translated_self, inputs_tgt_translated_self = None, None

        return outputs_dsc_src_translated, outputs_dsc_tgt_translated, \
                inputs_src_cycled, inputs_tgt_cycled, \
                    outputs_src_translated, outputs_tgt_translated, \
                        outputs_src, outputs_tgt, \
                            inputs_src_translated, inputs_tgt_translated, \
                                inputs_src_translated_self, inputs_tgt_translated_self

    def forward_dsc(self, inputs_src, inputs_tgt, p=1):
        inputs_tgt_translated = self.generator_st(inputs_src)
        inputs_src_translated = self.generator_ts(inputs_tgt)

        inputs_tgt_translated = (1 - p) * inputs_src + p * inputs_tgt_translated
        inputs_src_translated = (1 - p) * inputs_tgt + p * inputs_src_translated

        # Train discriminator
        outputs_dsc_src = self.discriminator_src(inputs_src)
        outputs_dsc_src_translated = self.discriminator_src(
            self.image_pool_src.query(inputs_src_translated)
        )

        outputs_dsc_tgt = self.discriminator_tgt(inputs_tgt)
        outputs_dsc_tgt_translated = self.discriminator_tgt(
            self.image_pool_tgt.query(inputs_tgt_translated)
        )

        return outputs_dsc_src, outputs_dsc_src_translated, \
            outputs_dsc_tgt, outputs_dsc_tgt_translated

    def get_parameters_generator(self):
        return list(self.generator_st.parameters()) + \
            list(self.generator_ts.parameters())

    def get_parameters_discriminator(self):
        return list(self.discriminator_src.parameters()) + \
            list(self.discriminator_tgt.parameters())

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

    def forward(self, inputs):
        x = self.block_down(inputs)
        x = self.blocks(x)
        x = self.block_up(x)

        return x


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
