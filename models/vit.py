import copy
import math
import torch
import torch.nn as nn

from mmcv.runner import _load_checkpoint
from mmseg.models import BACKBONES
from mmseg.models import VisionTransformer as _VisionTransformer
from mmseg.models.backbones.resnet import ResNet, Bottleneck, BasicBlock
from mmseg.utils import get_root_logger
from mmseg.ops import resize
from timm.models.vision_transformer_hybrid import HybridEmbed as _HybridEmbed, _resnetv2

from utils.checkpoint import convert_vit


@BACKBONES.register_module(name='VisionTransformer', force=True)
class VisionTransformer(_VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, embed_dims=768, deit=False, *args, **kwargs):
        super(VisionTransformer, self).__init__(
            img_size=img_size, patch_size=patch_size, embed_dims=embed_dims, *args, **kwargs)

        self.deit = deit
        if deit:
            # NOTE: initialization of dist_token is not supported yet (no need)
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
            num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + int(self.deit), embed_dims))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'model_ema' in checkpoint or 'model' in checkpoint:
                # deit pretrained model
                logger.info('Converting DeiT pretrained models..')
                state_dict = checkpoint['model_ema'] if 'model_ema' in checkpoint else checkpoint['model']
                state_dict = convert_vit(state_dict)
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed is None:
                    logger.warning('Found pos_embed in state_dict. However, the model does not have it.')
                elif self.pos_embed.shape != state_dict['pos_embed'].shape:
                    logger.info(msg=f'Resize the pos_embed shape from '
                                f'{state_dict["pos_embed"].shape} to '
                                f'{self.pos_embed.shape}')
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode, deit=self.deit)

            res = self.load_state_dict(state_dict, False)
            logger.info(res)

        elif self.pretrained is None:
            super(VisionTransformer, self).init_weights()

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        """Support distillation token."""
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size) + 1 + int(self.deit):
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.interpolate_mode, deit=self.deit)
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode, deit=False):
        """Support distillation token."""
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        dist_token_weight = pos_embed[:, 1:(-1 * pos_h * pos_w)]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        if deit:
            pos_embed = torch.cat((cls_token_weight, dist_token_weight, pos_embed_weight), dim=1)
        else:
            pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward(self, inputs):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs), (self.patch_embed.DH,
                                                 self.patch_embed.DW)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cats = (cls_tokens, self.dist_token.expand(B, -1, -1), x) if self.deit else (cls_tokens, x)
        x = torch.cat(cats, dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1 + int(self.deit):]

        outs = []
        if -1 in self.out_indices:
            if self.with_cls_token:
                # Remove class token and reshape token for decoder head
                out = x[:, 1 + int(self.deit):]
            else:
                out = x
            B, _, C = out.shape
            out = out.reshape(B, hw_shape[0], hw_shape[1],
                              C).permute(0, 3, 1, 2)
            if self.output_cls_token:
                out = [out, x[:, 0]]
            outs.append(out)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1 + int(self.deit):]
                else:
                    out = x
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2)
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs)


class HybridEmbed(_HybridEmbed):

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x)
        self.DH, self.DW = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x


class ResNetExt(ResNet):

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        26: (Bottleneck, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def forward(self, x):
        return super(ResNetExt, self).forward(x)[-1]


@BACKBONES.register_module()
class VisionTransformerHybrid(_VisionTransformer):

    def __init__(self, embed_type: str, img_size=224, patch_size=16, in_channels=3, embed_dims=768, use_origin=True, **kwargs):
        super(VisionTransformerHybrid, self).__init__(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dims=embed_dims, **kwargs)

        # we follow naming in timm
        if use_origin:
            # this is used to calculate GFLOPs, since some of the timm ops is not supported
            if embed_type == 'vit_small_r26_s32_384':
                stem_backbone = _resnetv2((2, 2, 2, 2))
            elif embed_type == 'vit_large_r50_s32_384':
                stem_backbone = _resnetv2((3, 4, 6, 3))
            elif embed_type == 'vit_base_r50_s16_384':
                stem_backbone = _resnetv2((3, 4, 9))
            else:
                raise NotImplementedError
        else:
            if embed_type == 'vit_small_r26_s32_384':
                stem_backbone = ResNetExt(depth=26, num_stages=4, out_indices=(3,))
            elif embed_type == 'vit_large_r50_s32_384':
                stem_backbone = ResNetExt(depth=50, num_stages=4, out_indices=(3,))
            else:
                raise NotImplementedError

        embed_layer = HybridEmbed(
            backbone=stem_backbone, img_size=img_size, patch_size=1,  # FIXME: do not apply to tiny hybrid model
            in_chans=in_channels, embed_dim=embed_dims
        )
        setattr(self, 'patch_embed', embed_layer)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    logger.info(msg=f'Resize the pos_embed shape from '
                                f'{state_dict["pos_embed"].shape} to '
                                f'{self.pos_embed.shape}')
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode)

            res = self.load_state_dict(state_dict, False)
            logger.info(res)

        elif self.pretrained is None:
            super(VisionTransformerHybrid, self).init_weights()

    def forward(self, inputs):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs), (self.patch_embed.DH,
                                                 self.patch_embed.DW)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        if -1 in self.out_indices:
            if self.with_cls_token:
                # Remove class token and reshape token for decoder head
                out = x[:, 1:]
            else:
                out = x
            B, _, C = out.shape
            out = out.reshape(B, hw_shape[0], hw_shape[1],
                              C).permute(0, 3, 1, 2)
            if self.output_cls_token:
                out = [out, x[:, 0]]
            outs.append(out)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2)
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs)