import torch
import torch.nn as nn
import torch.nn.functional as F

from module.etc import ImagePool
from pl_module.timm import LitTimm


class ZSGcGANModel(nn.Module):
    def __init__(self, config):
        super(ZSGcGANModel, self).__init__()

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

        self.discriminator_c = D_encoder()
        self.discriminator_b = D_encoder()

        # Image history replay
        self.image_pool = ImagePool(config.batch_size // 2)
        self.image_pool_gc = ImagePool(config.batch_size // 2)
        self.image_pool_c = ImagePool(config.batch_size // 2)
        self.image_pool_b = ImagePool(config.batch_size // 2)

        assert config.pretrained is not None
        self.pretrained = LitTimm.load_from_checkpoint(config.pretrained, config=config).model
        self.pretrained.eval()

    def forward(self, inputs_src):
        return self.pretrained(inputs_src)

    def forward_gan(self, inputs_src_toi, inputs_tgt_irt):
        inputs_tgt_irt_gc = self.transformation(inputs_tgt_irt, 0)

        # identity mapping with domain encoder
        inputs_src_toi_self_translated, _, x_src = self.generator_ts(inputs_src_toi, mode="b")
        _, x_tgt = self.generator_ts.forward_feature(inputs_tgt_irt, mode="b")

        # semantic identity mapping
        if self.identity_semantic_consistency:
            inputs_src_toi_self_translated_sem_idt, x_toi_self_translated_sem_idt, _ = self.generator_ts(inputs_src_toi, mode="c")
            outputs_src_toi_self_translated = self.pretrained(inputs_src_toi_self_translated_sem_idt)
            # outputs_src_toi = self.pretrained(inputs_src_toi)

            outputs_tgt_toi_syn = self.pretrained(self.generator_ts.get_image_from_cb(x_toi_self_translated_sem_idt, x_tgt.detach()))
        else:
            x_toi_self_translated_sem_idt, _ = self.generator_ts.forward_feature(inputs_src_toi, mode="c")
            outputs_src_toi_self_translated, outputs_tgt_toi_syn = None, None

        # geometric consistency with content encoder
        inputs_src_irt_translated, x_irt, _ = self.generator_ts(inputs_tgt_irt, mode="c")
        outputs_dsc_src_irt_translated = self.discriminator(inputs_src_irt_translated)

        inputs_src_irt_gc_translated, x_irt_gc, _ = self.generator_ts(inputs_tgt_irt_gc, mode="c")
        outputs_dsc_src_irt_gc_translated = self.discriminator_gc(inputs_src_irt_gc_translated)

        # fool encoder discriminator
        outputs_dsc_b = self.discriminator_b(x_irt)
        outputs_dsc_b_gc = self.discriminator_b(x_irt_gc)
        outputs_dsc_b_self_translated_sem_idt = self.discriminator_b(x_toi_self_translated_sem_idt)
        outputs_dsc_c_src = self.discriminator_c(x_src)
        outputs_dsc_c_tgt = self.discriminator_c(x_tgt)

        return inputs_src_irt_translated, self.transformation(inputs_src_irt_gc_translated, 1), \
            inputs_src_irt_gc_translated, self.transformation(inputs_src_irt_translated, 0), \
                outputs_dsc_src_irt_translated, outputs_dsc_src_irt_gc_translated, \
                    inputs_src_toi_self_translated, outputs_src_toi_self_translated, outputs_tgt_toi_syn, \
                        outputs_dsc_b, outputs_dsc_b_gc, outputs_dsc_c_src, outputs_dsc_c_tgt, \
                            outputs_dsc_b_self_translated_sem_idt

    def forward_dsc(self, inputs_src_toi, inputs_tgt_irt):
        inputs_tgt_irt_gc = self.transformation(inputs_tgt_irt, 0)

        inputs_src_irt_translated, x_irt, x_tgt = self.generator_ts(inputs_tgt_irt)
        inputs_src_irt_gc_translated, x_irt_gc, x_tgt_gc = self.generator_ts(inputs_tgt_irt_gc)
        x_toi, x_src = self.generator_ts.forward_feature(inputs_src_toi)

        # Train discriminator
        inputs_src_toi_gc = self.transformation(inputs_src_toi, 0)
        outputs_dsc_src_toi = self.discriminator(inputs_src_toi)
        outputs_dsc_src_irt_translated = self.discriminator(
            self.image_pool.query(inputs_src_irt_translated)
        )

        outputs_dsc_src_toi_gc = self.discriminator_gc(inputs_src_toi_gc)
        outputs_dsc_src_irt_gc_translated = self.discriminator_gc(
            self.image_pool_gc.query(inputs_src_irt_gc_translated)
        )

        outputs_dsc_irt = self.discriminator_b(self.image_pool_b.query(x_irt))
        outputs_dsc_irt_gc = self.discriminator_b(self.image_pool_b.query(x_irt_gc))
        outputs_dsc_toi = self.discriminator_b(self.image_pool_b.query(x_toi))
        outputs_dsc_tgt = self.discriminator_c(self.image_pool_c.query(x_tgt))
        outputs_dsc_tgt_gc = self.discriminator_c(self.image_pool_c.query(x_tgt_gc))
        outputs_dsc_src = self.discriminator_c(self.image_pool_c.query(x_src))

        return outputs_dsc_src_toi, outputs_dsc_src_irt_translated, \
            outputs_dsc_src_toi_gc, outputs_dsc_src_irt_gc_translated, \
                outputs_dsc_irt, outputs_dsc_irt_gc, outputs_dsc_toi, \
                    outputs_dsc_tgt, outputs_dsc_tgt_gc, outputs_dsc_src

    def translate(self, x, random_noise=None):
        return self.generator_ts(x, random_noise=random_noise)

    def get_feature(self, x):
        x_c, x_b = self.generator_ts.get_feature(x)
        return x_c, x_b

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

    def get_parameters_discriminator(self, learning_rate=None):
        if learning_rate is not None:
            return [
                {
                    "params": list(self.discriminator.parameters()) + \
                        list(self.discriminator_gc.parameters()),
                    "lr": learning_rate,
                },
                {
                    "params": list(self.discriminator_c.parameters()) + \
                        list(self.discriminator_b.parameters()),
                    "lr": learning_rate,
                },
            ]
        else:
            return list(self.discriminator.parameters()) + \
                list(self.discriminator_gc.parameters()) + \
                    list(self.discriminator_c.parameters()) + \
                        list(self.discriminator_b.parameters())

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

        # Content
        self.block_down_c = nn.Sequential(
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

        blocks_c = []
        for _ in range(num_blocks // 2):
            blocks_c.append(Block())
        self.blocks_c = nn.Sequential(*blocks_c)

        # Background
        self.block_down_b = nn.Sequential(
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

        blocks_b = []
        for _ in range(num_blocks // 2):
            blocks_b.append(Block())
        self.blocks_b = nn.Sequential(*blocks_b)

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

    def forward(self, inputs, mode="all"):
        x_c, x_b = self.forward_feature(inputs, mode)
        x = x_c + x_b

        x = self.block_up(x)

        return x, x_c, x_b

    def forward_feature(self, inputs, mode="all"):
        if mode == "b":
            with torch.no_grad():
                x_c = self.block_down_c(inputs)
                x_c = self.blocks_c(x_c)
        else:
            x_c = self.block_down_c(inputs)
            x_c = self.blocks_c(x_c)

        if mode == "c":
            with torch.no_grad():
                x_b = self.block_down_b(inputs)
                x_b = self.blocks_b(x_b)
        else:
            x_b = self.block_down_b(inputs)
            x_b = self.blocks_b(x_b)

        return x_c, x_b

    def get_feature(self, x):
        x_c = self.block_down_c(x)
        x_c = self.blocks_c(x_c)

        x_b = self.block_down_b(x)
        x_b = self.blocks_b(x_b)

        return x_c, x_b

    def get_image_from_cb(self, x_c, x_b):
        return self.block_up(x_c + x_b)


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


class D_encoder(nn.Module):
    def __init__(self):
        super(D_encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # DebugLayer(),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # DebugLayer(),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, x.size()[2:])
        return x.squeeze()


class DebugLayer(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x
