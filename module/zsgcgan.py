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
        self.discriminator_cross = D_encoder()

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
        inputs_src_toi_self_translated, x_toi, x_src = self.generator_ts(inputs_src_toi)

        # semantic identity mapping
        if self.identity_semantic_consistency:
            inputs_src_toi_self_translated_sem_idt, _, _ = self.generator_ts(inputs_src_toi)
            outputs_src_toi_self_translated = self.pretrained(inputs_src_toi_self_translated_sem_idt)
            outputs_src_toi = self.pretrained(inputs_src_toi)
        else:
            outputs_src_toi_self_translated, outputs_src_toi = None, None

        # geometric consistency with content encoder
        inputs_src_irt_translated, x_tgt, x_irt = self.generator_ts(inputs_tgt_irt)
        outputs_dsc_src_irt_translated = self.discriminator(inputs_src_irt_translated)

        inputs_src_irt_gc_translated, _, _ = self.generator_ts(inputs_tgt_irt_gc)
        outputs_dsc_src_irt_gc_translated = self.discriminator_gc(inputs_src_irt_gc_translated)

        outputs_dsc_cross_rec_src = self.discriminator_cross(x_toi + x_src)
        outputs_dsc_cross_rec_tgt = self.discriminator_cross(x_tgt + x_irt)

        outputs_dsc_cross_syn_src = self.discriminator_cross(x_toi + x_tgt)
        outputs_dsc_cross_syn_tgt = self.discriminator_cross(x_irt + x_src)

        return inputs_src_irt_translated, self.transformation(inputs_src_irt_gc_translated, 1), \
            inputs_src_irt_gc_translated, self.transformation(inputs_src_irt_translated, 0), \
                outputs_dsc_src_irt_translated, outputs_dsc_src_irt_gc_translated, \
                    inputs_src_toi_self_translated, outputs_src_toi_self_translated, outputs_src_toi, \
                        outputs_dsc_cross_rec_src, outputs_dsc_cross_rec_tgt, outputs_dsc_cross_syn_src, outputs_dsc_cross_syn_tgt

    def forward_dsc(self, inputs_src_toi, inputs_tgt_irt):
        inputs_tgt_irt_gc = self.transformation(inputs_tgt_irt, 0)

        inputs_src_irt_translated, x_irt, x_tgt = self.generator_ts(inputs_tgt_irt)
        inputs_src_irt_gc_translated, _, _ = self.generator_ts(inputs_tgt_irt_gc)

        inputs_src_toi_self_translated, x_toi, x_src = self.generator_ts(inputs_src_toi)

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

        outputs_dsc_cross_rec_src = self.discriminator_cross(x_toi + x_src)
        outputs_dsc_cross_rec_tgt = self.discriminator_cross(x_tgt + x_irt)

        outputs_dsc_cross_syn_src = self.discriminator_cross(x_toi + x_tgt)
        outputs_dsc_cross_syn_tgt = self.discriminator_cross(x_irt + x_src)

        return outputs_dsc_src_toi, outputs_dsc_src_irt_translated, \
            outputs_dsc_src_toi_gc, outputs_dsc_src_irt_gc_translated, \
                outputs_dsc_cross_rec_src, outputs_dsc_cross_rec_tgt, \
                    outputs_dsc_cross_syn_src, outputs_dsc_cross_syn_tgt

    def forward_enc_b(self, inputs_src_toi, inputs_tgt_irt):
        x_src = self.generator_ts.forward_feature(inputs_src_toi, enc_type="b")
        x_tgt = self.generator_ts.forward_feature(inputs_tgt_irt, enc_type="b")

        outputs_dsc_d_src = self.discriminator_b(x_src)
        outputs_dsc_d_tgt = self.discriminator_b(x_tgt)

        return outputs_dsc_d_src, outputs_dsc_d_tgt

    def forward_enc_c(self, inputs_src_toi, inputs_tgt_irt):
        x_toi = self.generator_ts.forward_feature(inputs_src_toi, enc_type="c")
        x_irt = self.generator_ts.forward_feature(inputs_tgt_irt, enc_type="c")

        outputs_dsc_c_toi = self.discriminator_c(x_toi)
        outputs_dsc_c_irt = self.discriminator_c(x_irt)

        return outputs_dsc_c_toi, outputs_dsc_c_irt

    def forward_cross(self, inputs_src_toi, inputs_tgt_irt):
        x_toi, x_src = self.generator_ts.forward_feature_all(inputs_src_toi)
        x_irt, x_tgt = self.generator_ts.forward_feature_all(inputs_tgt_irt)

        outputs_rec_src = self.discriminator_cross(self.generator_ts.forward_up(x_toi, x_src))
        outputs_rec_tgt = self.discriminator_cross(self.generator_ts.forward_up(x_irt, x_tgt))

        outputs_cross_src = self.discriminator_cross(self.generator_ts.forward_up(x_toi, x_src))
        outputs_cross_tgt = self.discriminator_cross(self.generator_ts.forward_up(x_irt, x_tgt))

        return outputs_rec_src, outputs_rec_tgt, \
            outputs_cross_src, outputs_cross_tgt

    def translate(self, x):
        translated, _, _ = self.generator_ts(x)
        return translated

    def get_feature(self, x):
        x_c, x_b = self.generator_ts.get_feature(x)
        return x_c, x_b

    def get_transformation(self, transformation):
        if transformation == "rotate":
            return self.rot90
        elif transformation == "flip":
            return self.vflip

    def rot90(self, tensor, direction):
        if direction == 0:
            return torch.rot90(tensor, -1, [-2, -1])
        else:
            return torch.rot90(tensor, 1, [-2, -1])

    def vflip(self, tensor, _):
        return torch.fliplr(tensor)

    def get_parameters_generator(self):
        return list(self.generator_ts.get_parameters()) + self.generator_ts.get_parameters_enc("all")

    def get_parameters_discriminator(self):
        return list(self.discriminator.parameters()) + \
            list(self.discriminator_gc.parameters()) + \
                list(self.discriminator_cross.parameters())

    def get_parameters_generator_enc(self, enc_type):
        return self.generator_ts.get_parameters_enc(enc_type)

    def get_parameters_discriminator_enc(self, enc_type):
        if enc_type == "c":
            return list(self.discriminator_c.parameters())
        elif enc_type == "b":
            return list(self.discriminator_b.parameters())
        else:
            raise NotImplementedError

    def get_parameters_generator_up(self):
        return self.generator_ts.block_up.parameters()

    def get_parameters_discriminator_up(self):
        return self.discriminator_cross.parameters()

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

        # self.mixer = nn.Sequential(
        #     nn.Linear(512, 256),
        #     # nn.ReLU(inplace=True),
        #     # nn.Linear(256, 256),
        #     # nn.ReLU(inplace=True),
        #     # nn.Linear(256, 256),
        # )

    def forward(self, inputs):
        x_c = self.block_down_c(inputs)
        x_c = self.blocks_c(x_c)

        x_b = self.block_down_b(inputs)
        x_b = self.blocks_b(x_b)

        x = self.forward_up(x_c, x_b)

        return x, x_c, x_b

    def forward_up(self, x_c, x_b):
        x = x_c + x_b
        # x = torch.cat([x_c, x_b], dim=1).permute(0,2,3,1)
        # x = self.mixer(x)
        # x = x.permute(0,3,1,2)

        x = self.block_up(x)

        return x

    def forward_feature(self, x, enc_type):
        if enc_type == "c":
            x = self.block_down_c(x)
            x = self.blocks_c(x)
        elif enc_type == "b":
            x = self.block_down_b(x)
            x = self.blocks_b(x)
        else:
            raise NotImplementedError
        return x

    def forward_feature_all(self, x):
        x_c = self.block_down_c(x)
        x_c = self.blocks_c(x_c)

        x_b = self.block_down_b(x)
        x_b = self.blocks_b(x_b)

        return x_c, x_b

    def get_parameters_enc(self, enc_type):
        if enc_type == "b":
            return list(self.block_down_b.parameters()) + \
                list(self.blocks_b.parameters())
        elif enc_type == "c":
            return list(self.block_down_c.parameters()) + \
                list(self.blocks_c.parameters())
        elif enc_type == "all":
            return list(self.block_down_c.parameters()) + \
                list(self.blocks_c.parameters()) + \
                    list(self.block_down_b.parameters()) + \
                        list(self.blocks_b.parameters())
        else:
            raise NotImplementedError

    def get_parameters(self):
        # return list(self.mixer.parameters()) + \
        #     list(self.block_up.parameters())
        # return self.parameters()
        return list(self.block_up.parameters())

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
