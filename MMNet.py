import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from Model.models.sam import sam_model_registry
from Model.models.dinov2 import dinov2_model_registry
import Model.cfg as cfg
from Model.models.ImageEncoder.vit.peft_modules import apply_mcrc_mask

class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y
    
class SEFusion(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SEFusion, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_dsm = SqueezeAndExcitation(channels_in,
                                           activation=activation)

    def forward(self, rgb, dsm):
        rgb = self.se_rgb(rgb)
        dsm = self.se_dsm(dsm)
        out = rgb + dsm
        return out

class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat

class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                ConvBNReLU(in_channels, out_channels, kernel_size=1),
            )
            for ps in pool_sizes
        ])
        self.bottleneck = ConvBNReLU(
            in_channels + len(pool_sizes) * out_channels,
            out_channels,
            kernel_size=3,
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        ppm_outs = [x]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
            ppm_outs.append(y)
        ppm_outs = torch.cat(ppm_outs, dim=1)
        return self.bottleneck(ppm_outs)


class UperNetHead(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        num_classes,
        dropout=0.1,
        pool_sizes=(1, 2, 3, 6),
    ):
        super().__init__()
        self.ppm = PPM(in_channels[-1], channels, pool_sizes)
        self.lateral_convs = nn.ModuleList([
            ConvBN(in_channels[i], channels, kernel_size=1)
            for i in range(len(in_channels) - 1)
        ])
        self.fpn_convs = nn.ModuleList([
            ConvBNReLU(channels, channels, kernel_size=3)
            for _ in range(len(in_channels) - 1)
        ])
        self.fpn_bottleneck = ConvBNReLU(
            channels * len(in_channels),
            channels,
            kernel_size=3,
        )
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(channels, num_classes, kernel_size=1),
        )

    def forward(self, feats, h, w):
        c1, c2, c3, c4 = feats
        p4 = self.ppm(c4)

        laterals = [
            self.lateral_convs[0](c1),
            self.lateral_convs[1](c2),
            self.lateral_convs[2](c3),
        ]

        p3 = laterals[2] + F.interpolate(p4, size=laterals[2].shape[2:], mode='bilinear', align_corners=False)
        p2 = laterals[1] + F.interpolate(p3, size=laterals[1].shape[2:], mode='bilinear', align_corners=False)
        p1 = laterals[0] + F.interpolate(p2, size=laterals[0].shape[2:], mode='bilinear', align_corners=False)

        fpn_outs = [
            self.fpn_convs[0](p1),
            self.fpn_convs[1](p2),
            self.fpn_convs[2](p3),
            p4,
        ]

        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=False,
            )

        fusion = torch.cat(fpn_outs, dim=1)
        out = self.fpn_bottleneck(fusion)
        out = self.classifier(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out

def draw_features(feature, savename=''):
    H = W = 256
    visualize = F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=False)
    visualize = visualize.detach().cpu().numpy()
    visualize = np.mean(visualize, axis=1).reshape(H, W)
    visualize = (((visualize - np.min(visualize)) / (np.max(visualize) - np.min(visualize))) * 255).astype(np.uint8)
    # fvis = np.fft.fft2(visualize)
    # fshift = np.fft.fftshift(fvis)
    # fshift = 20*np.log(np.abs(fshift))
    savedir = savename
    visualize = cv2.applyColorMap(visualize, cv2.COLORMAP_JET)
    cv2.imwrite(savedir, visualize)

class MMNet(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()
        args = cfg.parse_args()
        encoder_name = getattr(args, "encoder", "dinov2_vitl14")
        if encoder_name in dinov2_model_registry or encoder_name.startswith("dinov2"):
            build_fn = dinov2_model_registry.get(encoder_name, dinov2_model_registry["dinov2_vitl14"])
            dinov2_ckpt = getattr(args, "dinov2_ckpt", None)
            self.image_encoder = build_fn(args, checkpoint=dinov2_ckpt)
            self.sam = None
            is_dinov2 = True
        else:
            if encoder_name.startswith("sam_"):
                sam_key = encoder_name.replace("sam_", "")
            else:
                sam_key = encoder_name
            if sam_key not in sam_model_registry:
                sam_key = "vit_l"
            sam_ckpt = getattr(args, "sam_ckpt", None)
            if not sam_ckpt:
                if sam_key == "vit_b":
                    sam_ckpt = "weights/sam_vit_b_01ec64.pth"
                elif sam_key == "vit_h":
                    sam_ckpt = "weights/sam_vit_h_4b8939.pth"
                else:
                    sam_ckpt = "weights/sam_vit_l_0b3195.pth"
            self.sam = sam_model_registry[sam_key](args, checkpoint=sam_ckpt)
            self.image_encoder = self.sam.image_encoder
            is_dinov2 = False
        self.is_dinov2 = is_dinov2
        encoder_channels = (256, 256, 256, 256)
        self.cpia_enabled = getattr(self.image_encoder, "cpia_enabled", True)
        self.bagf_enabled = getattr(self.image_encoder, "bagf_enabled", True)
        if not self.is_dinov2:
            self.fpn1x = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                Norm2d(256),
                nn.GELU(),
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            )
            self.fpn2x = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            )
            self.fpn3x = nn.Identity()
            self.fpn4x = nn.MaxPool2d(kernel_size=2, stride=2)
            
            self.fpn1y = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            )
            self.fpn2y = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            )
            self.fpn3y = nn.Identity()
            self.fpn4y = nn.MaxPool2d(kernel_size=2, stride=2)
            
            self.fusion1 = SEFusion(encoder_channels[0])
            self.fusion2 = SEFusion(encoder_channels[1])
            self.fusion3 = SEFusion(encoder_channels[2])
            self.fusion4 = SEFusion(encoder_channels[3])
        for n, value in self.image_encoder.named_parameters():
            if is_dinov2:
                trainable = False
                if "dual_proj_layers" in n:
                    trainable = True
                elif "peft_cpia_blocks" in n:
                    trainable = self.cpia_enabled
                elif "peft_bagf_blocks" in n or "peft_fuse_norms" in n:
                    trainable = self.bagf_enabled
                elif n.startswith("extra_patch_embed.") or ".extra_patch_embed." in n:
                    trainable = getattr(self.image_encoder, "use_extra_patch_embed", False)
                value.requires_grad = trainable
                continue
            if n.startswith("patch_embed."):
                value.requires_grad = False
                continue
            if n.startswith("extra_patch_embed."):
                value.requires_grad = True
                continue
            if args.mod in ("sam_lora", "sam_adalora"):
                trainable = "lora_" in n
            elif args.mod == "sam_adpt":
                trainable = "Adapter" in n
            elif args.mod == "sam_peft":
                trainable = "peft_" in n
            else:
                trainable = True
            value.requires_grad = trainable

        self.decoder = UperNetHead(
            in_channels=encoder_channels,
            channels=decode_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.aux_head_rgb = AuxHead(in_channels=encoder_channels[0], num_classes=num_classes)
        self.aux_head_dsm = AuxHead(in_channels=encoder_channels[0], num_classes=num_classes)
        self.mcrc_enabled = getattr(args, "mcrc", False)
        self.mcrc_ratio = getattr(args, "mcrc_ratio", 0.5)
        self.mcrc_local_blocks = getattr(args, "mcrc_local_blocks", 1)
        self.mcrc_block_scale = (
            getattr(args, "mcrc_block_scale_min", 0.1),
            getattr(args, "mcrc_block_scale_max", 0.3),
        )
        self.mcrc_block_aspect = (
            getattr(args, "mcrc_block_aspect_min", 0.5),
            getattr(args, "mcrc_block_aspect_max", 2.0),
        )
        self.mcrc_aux_weight = getattr(args, "mcrc_aux_weight", 0.01)

    def forward(self, x, y, mode='Train'):
        h, w = x.size()[-2:]
        if mode == 'Train' and self.mcrc_enabled:
            x, y = apply_mcrc_mask(
                x,
                y,
                ratio=self.mcrc_ratio,
                num_blocks=self.mcrc_local_blocks,
                block_scale=self.mcrc_block_scale,
                aspect_ratio=self.mcrc_block_aspect,
            )
        y = torch.unsqueeze(y, dim=1).repeat(1,3,1,1)
        if self.is_dinov2:
            fused_feats, rgb_feats, dsm_feats = self.image_encoder(x, y)
            x = self.decoder(fused_feats, h, w)
            if mode == 'Train' and self.mcrc_enabled:
                x_rgb = self.aux_head_rgb(rgb_feats[0], h, w)
                x_dsm = self.aux_head_dsm(dsm_feats[0], h, w)
                return x, x_rgb, x_dsm
            return x

        deepx, deepy = self.image_encoder(x, y) # 256*16*16
        res1x = self.fpn1x(deepx)
        res2x = self.fpn2x(deepx)
        res3x = self.fpn3x(deepx)
        res4x = self.fpn4x(deepx)
        res1y = self.fpn1y(deepy)
        res2y = self.fpn2y(deepy)
        res3y = self.fpn3y(deepy)
        res4y = self.fpn4y(deepy)
        res1 = self.fusion1(res1x, res1y)
        res2 = self.fusion2(res2x, res2y)
        res3 = self.fusion3(res3x, res3y)
        res4 = self.fusion4(res4x, res4y)
        x = self.decoder([res1, res2, res3, res4], h, w)
        if mode == 'Train' and self.mcrc_enabled:
            x_rgb = self.aux_head_rgb(res1x, h, w)
            x_dsm = self.aux_head_dsm(res1y, h, w)
            return x, x_rgb, x_dsm
        return x 
