# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""
import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "DSConv",
    "ARConv"
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True, g=None):
        if g is None:
            g = c1  # 强制将 groups 设置为输入通道数 c1
        super().__init__(c1, c1, k, s, p=k // 2, d=d, g=g, act=act)  # 深度卷积：输入=输出=c1

class DSConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True, g=None):
        super().__init__()
        self.dwconv = DWConv(c1, c1, k, s, d, act, g)  # 深度卷积（groups=c1）
        self.pwconv = Conv(c1, c2, 1, 1, 0, act=act)  # 逐点卷积（c1 -> c2）
        self.stride = s
        self.padding = k // 2

    def forward(self, x):
        return self.pwconv(self.dwconv(x))

class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)



class ARConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, l_max=9, w_max=9, flag=False, modulation=True):
        super(ARConv, self).__init__()
        self.lmax = l_max
        self.wmax = w_max
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.flag = flag
        self.modulation = modulation
        self.i_list = [33, 35, 53, 37, 73, 55, 57, 75, 77]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(inc, outc, kernel_size=(i // 10, i % 10), stride=(i // 10, i % 10), padding=0)
                for i in self.i_list
            ]
        )
        self.m_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.Tanh()
        )
        self.b_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride)
        )
        self.p_conv = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(),
        )
        self.l_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.w_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout2d(0.3)
    #     self.hook_handles = []
    #     self.hook_handles.append(self.m_conv[0].register_full_backward_hook(self._set_lr))
    #     self.hook_handles.append(self.m_conv[1].register_full_backward_hook(self._set_lr))
    #     self.hook_handles.append(self.b_conv[0].register_full_backward_hook(self._set_lr))
    #     self.hook_handles.append(self.b_conv[1].register_full_backward_hook(self._set_lr))
    #     self.hook_handles.append(self.p_conv[0].register_full_backward_hook(self._set_lr))
    #     self.hook_handles.append(self.p_conv[1].register_full_backward_hook(self._set_lr))
    #     self.hook_handles.append(self.l_conv[0].register_full_backward_hook(self._set_lr))
    #     self.hook_handles.append(self.l_conv[1].register_full_backward_hook(self._set_lr))
    #     self.hook_handles.append(self.w_conv[0].register_full_backward_hook(self._set_lr))
    #     self.hook_handles.append(self.w_conv[1].register_full_backward_hook(self._set_lr))
 
        self.reserved_NXY = nn.Parameter(torch.tensor([3, 3], dtype=torch.int32), requires_grad=False)
 
    # @staticmethod
    # def _set_lr(module, grad_input, grad_output):
    #     grad_input = tuple(g * 0.1 if g is not None else None for g in grad_input)
    #     grad_output = tuple(g * 0.1 if g is not None else None for g in grad_output)
    #     return grad_input
 
    # def remove_hooks(self):
    #     for handle in self.hook_handles:
    #         handle.remove()  # 移除钩子函数
    #     self.hook_handles.clear()  # 清空句柄列表
 
    def forward(self, x, epoch, hw_range):
            # 使用全局变量作为默认值
        if epoch is None:
            epoch = TrainingState.epoch
        if hw_range is None:
            hw_range = TrainingState.hw_range

    # 可选：支持外部手动控制冻结
        if TrainingState.freeze_arconv:
            epoch = 999  # 直接进入冻结逻辑
        assert isinstance(hw_range, list) and len(hw_range) == 2, "hw_range should be a list with 2 elements, represent the range of h w"
        scale = hw_range[1] // 9
        if hw_range[0] == 1 and hw_range[1] == 3:
            scale = 1
        m = self.m_conv(x)
        bias = self.b_conv(x)
        offset = self.p_conv(x * 100)
        l = self.l_conv(offset) * (hw_range[1] - 1) + 1  # b, 1, h, w
        w = self.w_conv(offset) * (hw_range[1] - 1) + 1  # b, 1, h, w
        if epoch <= 100:
            mean_l = l.mean(dim=0).mean(dim=1).mean(dim=1)
            mean_w = w.mean(dim=0).mean(dim=1).mean(dim=1)
            N_X = int(mean_l // scale)
            N_Y = int(mean_w // scale)
            def phi(x):
                if x % 2 == 0:
                    x -= 1
                return x
            N_X, N_Y = phi(N_X), phi(N_Y)
            N_X, N_Y = max(N_X, 3), max(N_Y, 3)
            N_X, N_Y = min(N_X, 7), min(N_Y, 7)
            if epoch == 100:
                self.reserved_NXY = self.reserved_NXY = nn.Parameter(
                    torch.tensor([N_X, N_Y], dtype=torch.int32, device=x.device),
                    requires_grad=False
                )
        else:
            N_X = self.reserved_NXY[0]
            N_Y = self.reserved_NXY[1]

        N = N_X * N_Y
        # print(N_X, N_Y)
        l = l.repeat([1, N, 1, 1])
        w = w.repeat([1, N, 1, 1])
        offset = torch.cat((l, w), dim=1)
        dtype = offset.data.type()
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype, N_X, N_Y)  # (b, 2*N, h, w)
        p = p.contiguous().permute(0, 2, 3, 1)  # (b, h, w, 2*N)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        # clip p
        p = torch.cat(
            [
                torch.clamp(p[..., :N], 0, x.size(2) - 1),
                torch.clamp(p[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        )
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_lt[..., N:].type_as(p) - p[..., N:])
        )
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_rb[..., N:].type_as(p) - p[..., N:])
        )
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_lb[..., N:].type_as(p) - p[..., N:])
        )
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_rt[..., N:].type_as(p) - p[..., N:])
        )
        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        # (b, c, h, w, N)
        x_offset = (
                g_lt.unsqueeze(dim=1) * x_q_lt
                + g_rb.unsqueeze(dim=1) * x_q_rb
                + g_lb.unsqueeze(dim=1) * x_q_lb
                + g_rt.unsqueeze(dim=1) * x_q_rt
        )
        x_offset = self._reshape_x_offset(x_offset, N_X, N_Y)
        x_offset = self.dropout2(x_offset)
        x_offset = self.convs[self.i_list.index(N_X * 10 + N_Y)](x_offset)
        out = x_offset * m + bias
        return out
 
    def _get_p_n(self, N, dtype, n_x, n_y):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(n_x - 1) // 2, (n_x - 1) // 2 + 1),
            torch.arange(-(n_y - 1) // 2, (n_y - 1) // 2 + 1),
        )
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n
 
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0
 
    def _get_p(self, offset, dtype, n_x, n_y):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        L, W = offset.split([N, N], dim=1)
        L = L / n_x
        W = W / n_y
        offsett = torch.cat([L, W], dim=1)
        p_n = self._get_p_n(N, dtype, n_x, n_y)
        p_n = p_n.repeat([1, 1, h, w])
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + offsett * p_n
        return p
 
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = (
            index.contiguous()
            .unsqueeze(dim=1)
            .expand(-1, c, -1, -1, -1)
            .contiguous()
            .view(b, c, -1)
        )
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset
 
    @staticmethod
    def _reshape_x_offset(x_offset, n_x, n_y):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + n_y].contiguous().view(b, c, h, w * n_y) for s in range(0, N, n_y)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * n_x, w * n_y)
        return x_offset

# import torch
# import torch.nn as nn
# from train_state import TrainingState  # 你已定义的训练状态

# class ARConv(nn.Module):
#     def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, l_max=9, w_max=9, modulation=True):
#         super(ARConv, self).__init__()
#         self.lmax = l_max
#         self.wmax = w_max
#         self.inc = inc
#         self.outc = outc
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.stride = stride
#         self.zero_padding = nn.ZeroPad2d(padding)
#         self.modulation = modulation
#         self.i_list = [33, 35, 53, 37, 73, 55, 57, 75, 77]

#         self.convs = nn.ModuleList([
#             nn.Conv2d(inc, outc, kernel_size=(i // 10, i % 10), stride=(i // 10, i % 10), padding=0)
#             for i in self.i_list
#         ])

#         self.m_conv = nn.Sequential(
#             nn.Conv2d(inc, outc, 3, padding=1, stride=stride),
#             nn.LeakyReLU(0.1, inplace=False),
#             nn.Dropout2d(0.3),
#             nn.Conv2d(outc, outc, 3, padding=1, stride=stride),
#             nn.LeakyReLU(0.1, inplace=False),
#             nn.Dropout2d(0.3),
#             nn.Conv2d(outc, outc, 3, padding=1, stride=stride),
#             nn.Tanh()
#         )

#         self.b_conv = nn.Sequential(
#             nn.Conv2d(inc, outc, 3, padding=1, stride=stride),
#             nn.LeakyReLU(0.1, inplace=False),
#             nn.Dropout2d(0.3),
#             nn.Conv2d(outc, outc, 3, padding=1, stride=stride),
#             nn.LeakyReLU(0.1, inplace=False),
#             nn.Dropout2d(0.3),
#             nn.Conv2d(outc, outc, 3, padding=1, stride=stride)
#         )

#         self.p_conv = nn.Sequential(
#             nn.Conv2d(inc, inc, 3, padding=1, stride=stride),
#             nn.BatchNorm2d(inc),
#             nn.LeakyReLU(0.1, inplace=False),
#             nn.Conv2d(inc, inc, 3, padding=1, stride=stride),
#             nn.BatchNorm2d(inc),
#             nn.LeakyReLU(0.1, inplace=False),
#         )

#         self.l_conv = nn.Sequential(
#             nn.Conv2d(inc, 1, 3, padding=1, stride=stride),
#             nn.BatchNorm2d(1),
#             nn.LeakyReLU(0.1, inplace=False),
#             nn.Conv2d(1, 1, 1),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )

#         self.w_conv = nn.Sequential(
#             nn.Conv2d(inc, 1, 3, padding=1, stride=stride),
#             nn.BatchNorm2d(1),
#             nn.LeakyReLU(0.1, inplace=False),
#             nn.Conv2d(1, 1, 1),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )

#         self.dropout2 = nn.Dropout2d(0.3)
#         self.reserved_NXY = nn.Parameter(torch.tensor([3, 3], dtype=torch.int32), requires_grad=False)

#     def forward(self, x, epoch=None, hw_range=None):
#         if epoch is None:
#             epoch = TrainingState.epoch
#         if hw_range is None:
#             hw_range = TrainingState.hw_range
#         if TrainingState.freeze_arconv:
#             epoch = 999

#         scale = hw_range[1] // 9
#         m = self.m_conv(x)
#         bias = self.b_conv(x)
#         offset = self.p_conv(x * 100)
#         l = self.l_conv(offset) * (hw_range[1] - 1) + 1
#         w = self.w_conv(offset) * (hw_range[1] - 1) + 1

#         if epoch <= 100:
#             mean_l = l.mean()
#             mean_w = w.mean()
#             N_X = max(min((int(mean_l.item() / scale) | 1), 7), 3)  # 保证奇数，限制在3~7
#             N_Y = max(min((int(mean_w.item() / scale) | 1), 7), 3)
#             if epoch == 100:
#                 self.reserved_NXY = nn.Parameter(torch.tensor([N_X, N_Y], dtype=torch.int32, device=x.device), requires_grad=False)
#         else:
#             N_X, N_Y = self.reserved_NXY[0].item(), self.reserved_NXY[1].item()

#         N = N_X * N_Y
#         l = l.repeat([1, N, 1, 1])
#         w = w.repeat([1, N, 1, 1])
#         offset = torch.cat((l, w), dim=1)
#         if self.padding:
#             x = self.zero_padding(x)

#         p = self._get_p(offset, offset.data.type(), N_X, N_Y)  # (b, 2*N, h, w)
#         p = p.contiguous().permute(0, 2, 3, 1)  # (b, h, w, 2*N)
#         q_lt, q_rb = p.floor(), p.floor() + 1
#         q_lt = torch.cat([q_lt[..., :N].clamp(0, x.size(2)-1), q_lt[..., N:].clamp(0, x.size(3)-1)], dim=-1).long()
#         q_rb = torch.cat([q_rb[..., :N].clamp(0, x.size(2)-1), q_rb[..., N:].clamp(0, x.size(3)-1)], dim=-1).long()
#         q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
#         q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
#         p = torch.cat([p[..., :N].clamp(0, x.size(2)-1), p[..., N:].clamp(0, x.size(3)-1)], dim=-1)

#         g = lambda q, p: (1 - (q.type_as(p) - p)).abs()
#         g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
#         g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
#         g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
#         g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

#         x_q = lambda q: self._get_x_q(x, q, N)
#         x_offset = g_lt.unsqueeze(1)*x_q(q_lt) + g_rb.unsqueeze(1)*x_q(q_rb) + g_lb.unsqueeze(1)*x_q(q_lb) + g_rt.unsqueeze(1)*x_q(q_rt)
#         x_offset = self._reshape_x_offset(x_offset, N_X, N_Y)
#         x_offset = self.dropout2(x_offset)
#         x_offset = self.convs[self.i_list.index(N_X * 10 + N_Y)](x_offset)
#         return x_offset * m + bias

#     def _get_p_n(self, N, dtype, n_x, n_y):
#         p_n_x, p_n_y = torch.meshgrid(torch.arange(-(n_x - 1) // 2, (n_x - 1) // 2 + 1),
#                                       torch.arange(-(n_y - 1) // 2, (n_y - 1) // 2 + 1), indexing='ij')
#         p_n = torch.cat([p_n_x.flatten(), p_n_y.flatten()], 0).view(1, 2*N, 1, 1).type(dtype)
#         return p_n

#     def _get_p_0(self, h, w, N, dtype):
#         p_0_x, p_0_y = torch.meshgrid(torch.arange(1, h * self.stride + 1, self.stride),
#                                       torch.arange(1, w * self.stride + 1, self.stride), indexing='ij')
#         p_0 = torch.cat([
#             p_0_x.flatten().view(1, 1, h, w).repeat(1, N, 1, 1),
#             p_0_y.flatten().view(1, 1, h, w).repeat(1, N, 1, 1)
#         ], dim=1).type(dtype)
#         return p_0

#     def _get_p(self, offset, dtype, n_x, n_y):
#         N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
#         L, W = offset.split([N, N], dim=1)
#         offset = torch.cat([L / n_x, W / n_y], dim=1)
#         return self._get_p_0(h, w, N, dtype) + offset * self._get_p_n(N, dtype, n_x, n_y).repeat(1, 1, h, w)

#     def _get_x_q(self, x, q, N):
#         b, h, w, _ = q.size()
#         c = x.size(1)
#         x = x.view(b, c, -1)
#         idx = q[..., :N]*x.size(3) + q[..., N:]
#         idx = idx.unsqueeze(1).expand(-1, c, -1, -1, -1).reshape(b, c, -1)
#         return x.gather(-1, idx).view(b, c, h, w, N)

#     def _reshape_x_offset(self, x_offset, n_x, n_y):
#         b, c, h, w, N = x_offset.size()
#         x_offset = torch.cat([
#             x_offset[..., i:i+n_y].contiguous().view(b, c, h, w * n_y)
#             for i in range(0, N, n_y)
#         ], dim=-1)
#         return x_offset.view(b, c, h * n_x, w * n_y)
