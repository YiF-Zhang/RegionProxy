import einops
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, trunc_normal_init
from mmcv.utils import to_2tuple


@HEADS.register_module()
class ProxyHead(BaseDecodeHead):

    def __init__(self, in_channels, channels, num_classes, region_res=(4, 4), norm_cfg=None, act_cfg=dict(type='ReLU'),
                 init_cfg={}, *args, **kwargs):

        super(ProxyHead, self).__init__(
            in_channels, channels, num_classes=num_classes, norm_cfg=norm_cfg,
            act_cfg=act_cfg, init_cfg=init_cfg, *args, **kwargs)

        self.region_res = to_2tuple(region_res)

        self.mlp = nn.Sequential(nn.Sequential(nn.Linear(in_channels[-1], num_classes)))
        self.affinity_head = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels[0], channels, kernel_size=3, padding=1, act_cfg=act_cfg, norm_cfg=norm_cfg),
            ConvModule(
                channels, 9 * self.region_res[0] * self.region_res[1], kernel_size=1, act_cfg=None)
        )

        delattr(self, 'conv_seg')

    def init_weights(self):
        super(ProxyHead, self).init_weights()
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=.0)
        assert all(self.affinity_head[-1].conv.bias == 0)

    def forward_affinity(self, x):
        self._device = x.device
        B, _, H, W = x.shape

        # get affinity
        x = x.contiguous()
        affinity = self.affinity_head(x)
        affinity = affinity.reshape(B, 9, *self.region_res, H, W)  # (B, 9, h, w, H, W)

        # handle borders
        affinity[:, :3, :, :, 0, :] = float('-inf')  # top
        affinity[:, -3:, :, :, -1, :] = float('-inf')  # bottom
        affinity[:, ::3, :, :, :, 0] = float('-inf')  # left
        affinity[:, 2::3, :, :, :, -1] = float('-inf')  # right

        affinity = affinity.softmax(dim=1)
        return affinity

    def forward_cls(self, x):
        self._device = x.device
        B, _, H, W = x.shape

        # get token logits
        token_logits = self.mlp(x.permute(0, 2, 3, 1).reshape(B, H * W, -1))  # (B, H * W, C)
        return token_logits

    def forward(self, inputs):
        x_mid, x = self._transform_inputs(inputs)  # (B, C, H, W)
        B, _, H, W = x.shape

        affinity = self.forward_affinity(x_mid)
        token_logits = self.forward_cls(x)

        # classification per pixel
        token_logits = token_logits.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, C, H, W)
        token_logits = F.unfold(token_logits, kernel_size=3, padding=1).reshape(B, -1, 9, H, W)  # (B, C, 9, H, W)
        token_logits = einops.rearrange(token_logits, 'B C n H W -> B H W n C')  # (B, H, W, 9, C)

        affinity = einops.rearrange(affinity, 'B n h w H W -> B H W (h w) n')  # (B, H, W, h * w, 9)
        seg_logits = (affinity @ token_logits).reshape(B, H, W, *self.region_res, -1)  # (B, H, W, h, w, C)
        seg_logits = einops.rearrange(seg_logits, 'B H W h w C -> B C (H h) (W w)')  # (B, C, H * h, W * w)

        return seg_logits
