import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from torch.jit import Final
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Union, List
from einops import rearrange, reduce
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv,ARConv, autopad
from ..modules.block import *

from .shiftwise_conv import ReparamLargeKernelConv

from .hcfnet import PPA, LocalGlobalAttention

from .deconv import DEConv

from .transformer import DynamicTanh

from ultralytics.utils.ops import make_divisible
from timm.layers import CondConv2d, trunc_normal_, use_fused_attn, to_2tuple
from timm.models import named_apply

__all__ = ['CSP_SPDP', 'HAFB']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



#结合PPA,增加注意力分支，丰富特征获取能力
class SPDP(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.inc = inc
        self.conv1 = ReparamLargeKernelConv(inc, inc, kernel_size=7, small_kernel=3, groups=inc // 8)
        self.conv2 = ReparamLargeKernelConv(inc // 2, inc // 2, kernel_size=13, small_kernel=5, groups=inc // 2)
        self.conv3 = ReparamLargeKernelConv(inc // 4, inc // 4, kernel_size=9, small_kernel=3, groups=inc // 4)
        self.conv4 = Conv(inc, inc, 1)

        # 注意力并行分支
        self.ppa_branch = PPA(inc, inc)

        # 门控机制：输入拼接后 shape = (B, 3*inc, H, W)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2 * inc, 2, kernel_size=1, stride=1),  # 输出2个通道，对应main/attn
            nn.Sigmoid()
        )

        # 融合后再压缩回 inc 通道
        self.fusion_conv = Conv(inc, inc, 1)

    def forward(self, x):
        # 主干结构提取
        conv1_out = self.conv1(x)
        conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
        conv2_out = self.conv2(conv1_out_1)#C/2
        conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)#C/4
        conv3_out = self.conv3(conv2_out_1)#C/4
        main_out = self.conv4(torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1))  # (B, inc, H, W)

        # 注意力分支
        attn_out = self.ppa_branch(x)  # (B, inc, H, W)

        # 三分支拼接用于门控计算
        concat_feat = torch.cat([main_out, attn_out], dim=1)  # (B, 3*inc, H, W)
        gates = self.gate_conv(concat_feat)  # (B, 3, H, W)

        # 拆分 gate 权重
        gate_main, gate_attn = gates[:, 0:1], gates[:, 1:2]

        # 加权融合
        fused = main_out * gate_main + attn_out * gate_attn 

        # 通道压缩融合
        out = self.fusion_conv(fused) + x # 输出仍是 (B, inc, H, W)

        return out
class CSP_SPDP(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(SPDP(self.c) for _ in range(n))


######################################## Hierarchical Attention Fusion Block start ########################################

class HAFB(nn.Module):
    # HAFB: Hierarchical Attention Fusion Block (层级注意力融合模块)
    # 该模块旨在融合两个不同尺度的输入特征图 (x1, x2)。
    def __init__(self, inc, ouc, group=False):
        """
        初始化层级注意力融合模块。
        Args:
            inc (tuple): 一个包含两个整数的元组，分别代表两个输入特征图的通道数 (ch_1, ch_2)。
            ouc (int): 输出特征图的通道数。
            group (bool): 一个布尔值，用于确定 RepConv 中是否使用分组卷积。
        """
        super(HAFB, self).__init__()
        # 解包输入通道数
        ch_1, ch_2 = inc
        # 定义一个中间隐藏层的通道数，为输出通道数的一半
        hidc = ouc // 2

        # --- 为第一个输入 (x1) 定义注意力分支 ---
        # 局部注意力分支，使用 2x2 的图块大小
        self.lgb1_local = LocalGlobalAttention(hidc, 2)
        # 全局注意力分支，使用 4x4 的图块大小
        self.lgb1_global = LocalGlobalAttention(hidc, 4)

        # --- 为第二个输入 (x2) 定义注意力分支 ---
        # 局部注意力分支，使用 2x2 的图块大小
        self.lgb2_local = LocalGlobalAttention(hidc, 2)
        # 全局注意力分支，使用 4x4 的图块大小
        self.lgb2_global = LocalGlobalAttention(hidc, 4)

        # --- 定义卷积层 ---
        # 使用1x1卷积调整第一个输入的通道数至 hidc
        self.W_x1 = Conv(ch_1, hidc, 1, act=False)
        # 使用1x1卷积调整第二个输入的通道数至 hidc
        self.W_x2 = Conv(ch_2, hidc, 1, act=False)
        # 用于基础路径 (basic path) 的卷积层
        self.W = Conv(hidc, ouc, 3, g=4) # g=4 表示分组卷积

        # 用于在拼接后压缩通道数的1x1卷积
        self.conv_squeeze = Conv(ouc * 3, ouc, 1)
        # RepConv层，用于增强特征表示
        self.rep_conv = RepConv(ouc, ouc, 3, g=(16 if group else 1))
        # 最终输出前的1x1卷积
        self.conv_final = Conv(ouc, ouc, 1)

    def forward(self, inputs):
        """
        HAFB模块的前向传播。
        Args:
            inputs (tuple): 包含两个输入特征图 (x1, x2) 的元组。
        Returns:
            torch.Tensor: 融合后的输出特征图。
        """
        # 解包两个输入特征图
        x1, x2 = inputs
        
        # 1. 对两个输入进行通道调整，使其通道数均为 hidc
        W_x1 = self.W_x1(x1)
        W_x2 = self.W_x2(x2)
        
        # 2. 计算基础路径 (Basic Path, bp)
        # 将调整通道后的两个特征图逐元素相加，然后通过卷积层
        bp = self.W(W_x1 + W_x2)

        # 3. 计算第一个输入的多尺度注意力特征
        # 将局部注意力输出和全局注意力输出沿通道维度拼接
        x1 = torch.cat([self.lgb1_local(W_x1), self.lgb1_global(W_x1)], dim=1)
        
        # 4. 计算第二个输入的多尺度注意力特征
        # 将局部注意力输出和全局注意力输出沿通道维度拼接
        x2 = torch.cat([self.lgb2_local(W_x2), self.lgb2_global(W_x2)], dim=1)

        # 5. 融合所有路径的特征并输出
        # a. 将两个注意力路径的输出 (x1, x2) 和基础路径的输出 (bp) 沿通道维度拼接
        # b. 使用 conv_squeeze 压缩通道数
        # c. 通过 RepConv 层进行特征增强
        # d. 通过最后的 conv_final 得到最终输出
        return self.conv_final(self.rep_conv(self.conv_squeeze(torch.cat([x1, x2, bp], 1))))

