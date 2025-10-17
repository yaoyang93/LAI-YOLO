import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ReparamLargeKernelConv']

def get_conv2d(
        in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
):
    """
    一个创建 nn.Conv2d 层的辅助函数。
    它包含一个try-except块来处理kernel_size为元组时自动计算padding的情况。
    """
    try:
        # 尝试将kernel_size视为元组(kh, kw)，并计算对应的padding
        paddings = (kernel_size[0] // 2, kernel_size[1] // 2)
    except Exception as e:
        # 如果失败（例如kernel_size是整数），则使用传入的padding值
        paddings = padding
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, paddings, dilation, groups, bias
    )

def get_bn(channels):
    """一个创建 nn.BatchNorm2d 层的辅助函数。"""
    return nn.BatchNorm2d(channels)

class Mask(nn.Module):
    """
    一个可学习的掩码模块。
    它创建一个与输入大小相同的可学习张量，通过Sigmoid函数将其值缩放到[0, 1]范围，
    然后与输入张量逐元素相乘，从而实现对输入的加权。
    """
    def __init__(self, size):
        super().__init__()
        # 创建一个可学习的参数张量，并在[-1, 1]范围内进行均匀初始化
        self.weight = torch.nn.Parameter(data=torch.Tensor(*size), requires_grad=True)
        self.weight.data.uniform_(-1, 1)

    def forward(self, x):
        # 将可学习权重通过Sigmoid函数生成掩码
        w = torch.sigmoid(self.weight)
        # 将掩码与输入相乘
        masked_wt = w.mul(x)
        return masked_wt

def conv_bn_ori(
        in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True
):
    """
    创建一个标准的 '卷积层 + (可选的)批归一化层' 序列。
    这是用于对称卷积核的基础构建块。
    """
    if padding is None:
        # 如果未提供padding，则自动计算以保持特征图大小不变
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module(
        "conv",
        get_conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False, # BN层会处理偏置，所以卷积层不需要
        ),
    )

    if bn:
        # 如果需要，添加批归一化层
        result.add_module("bn", get_bn(out_channels))
    return result

class LoRAConvsByWeight(nn.Module):
    """
    通过权重合并LoRA1和LoRA2。
    这个类实现了一种方法，使用多个小卷积核来模拟一个大的非对称卷积核。
    它通过对小卷积的输出进行特定的重排和相加来实现这一目的。
    'LoRA' 在此借指其分解和组合的思想。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 big_kernel, small_kernel,
                 stride=1, group=1,
                 bn=True, use_small_conv=True):
        super().__init__()
        self.kernels = (small_kernel, big_kernel)
        self.stride = stride
        self.small_conv = use_small_conv
        # 调用shift方法预计算重排所需的padding和索引
        padding, after_padding_index, index = self.shift(self.kernels)
        self.pad = padding, after_padding_index, index
        # 计算需要多少个小卷积核来覆盖一个大卷积核
        self.nk = math.ceil(big_kernel / small_kernel)
        # 计算总的输出通道数，用于单个大卷积层
        out_n = out_channels * self.nk
        # 定义一个统一的卷积层，它一次性计算出所有需要重排的特征
        self.split_convs = nn.Conv2d(in_channels, out_n,
                                     kernel_size=small_kernel, stride=stride,
                                     padding=padding, groups=group,
                                     bias=False)

        # 为两个方向（水平和垂直）的重组路径定义可学习的掩码
        self.lora1 = Mask((1, out_n, 1, 1))
        self.lora2 = Mask((1, out_n, 1, 1))
        self.use_bn = bn

        if bn:
            # 为两条路径分别定义批归一化层
            self.bn_lora1 = get_bn(out_channels)
            self.bn_lora2 = get_bn(out_channels)
        else:
            self.bn_lora1 = None
            self.bn_lora2 = None

    def forward(self, inputs):
        # 首先，通过统一的卷积层计算出所有基础特征
        out = self.split_convs(inputs)
        # 获取原始输入的尺寸，用于后续恢复
        *_, ori_h, ori_w = inputs.shape
        # 处理第一条LoRA路径（例如，水平方向'H'）
        lora1_x = self.forward_lora(self.lora1(out), ori_h, ori_w, VH='H', bn=self.bn_lora1)
        # 处理第二条LoRA路径（例如，垂直方向'W'）
        lora2_x = self.forward_lora(self.lora2(out), ori_h, ori_w, VH='W', bn=self.bn_lora2)
        # 将两条路径的结果相加
        x = lora1_x + lora2_x
        return x

    def forward_lora(self, out, ori_h, ori_w, VH='H', bn=None):
        """处理单条LoRA路径（水平或垂直）的前向传播"""
        b, c, h, w = out.shape
        # ※※※※※ 核心步骤 ※※※※※
        # 将输出张量重塑并分割，分离出对应 nk 个小卷积核的特征组
        out = torch.split(out.reshape(b, -1, self.nk, h, w), 1, 2)
        x = 0
        # 遍历每个特征组
        for i in range(self.nk):
            # 对每个特征组进行数据重排（空间上的平移和裁剪）
            outi = self.rearrange_data(out[i], i, ori_h, ori_w, VH)
            # 将重排后的结果累加起来
            x = x + outi
        if self.use_bn:
            # 应用批归一化
            x = bn(x)
        return x

    def rearrange_data(self, x, idx, ori_h, ori_w, VH):
        """对特征图进行空间重排，模拟大卷积核中的特定位置"""
        padding, _, index = self.pad
        # ※※※※※※※
        x = x.squeeze(2) # 移除之前分割时产生的维度
        *_, h, w = x.shape
        k = min(self.kernels)       # 小卷积核尺寸
        ori_k = max(self.kernels)   # 大卷积核尺寸
        ori_p = ori_k // 2
        stride = self.stride
        
        # 计算当前块在模拟大卷积核时需要的平移量和填充量
        if (idx + 1) >= index:
            pad_l = 0
            s = (idx + 1 - index) * (k // stride)
        else:
            pad_l = (index - 1 - idx) * (k // stride)
            s = 0
            
        if VH == 'H': # 水平方向
            # 计算理论上大卷积核应输出的特征图宽度
            suppose_len = (ori_w + 2 * ori_p - ori_k) // stride + 1
            pad_r = 0 if (s + suppose_len) <= (w + pad_l) else s + suppose_len - w - pad_l
            new_pad = (pad_l, pad_r, 0, 0) # (left, right, top, bottom)
            dim = 3 # 宽度维度
        else: # 垂直方向
            # 计算理论上大卷积核应输出的特征图高度
            suppose_len = (ori_h + 2 * ori_p - ori_k) // stride + 1
            pad_r = 0 if (s + suppose_len) <= (h + pad_l) else s + suppose_len - h - pad_l
            new_pad = (0, 0, pad_l, pad_r) # (left, right, top, bottom)
            dim = 2 # 高度维度
        
        # 如果计算出需要填充，则执行填充操作
        if len(set(new_pad)) > 1:
            x = F.pad(x, new_pad)
        
        # 处理padding不对称的情况，进行裁剪
        if padding * 2 + 1 != k:
            pad = padding - k // 2
            if VH == 'H':  # 水平方向
                x = torch.narrow(x, 2, pad, h - 2 * pad)
            else:  # 垂直方向
                x = torch.narrow(x, 3, pad, w - 2 * pad)
        
        # 使用torch.narrow进行切片，模拟平移操作
        xs = torch.narrow(x, dim, s, suppose_len)
        return xs

    def shift(self, kernels):
        """
        预计算重排所需的padding和索引。
        这是一个复杂的初始化步骤，其核心思想是计算出如何用小卷积核的输出来拼接成
        一个与大卷积核输出在空间上对齐的特征图。
        """
        mink, maxk = min(kernels), max(kernels)
        mid_p = maxk // 2
        offset_idx_left = mid_p % mink
        offset_idx_right = (math.ceil(maxk / mink) * mink - mid_p - 1) % mink
        padding = offset_idx_left % mink
        while padding < offset_idx_right:
            padding += mink
        while padding < (mink - 1):
            padding += mink
        after_padding_index = padding - offset_idx_left
        index = math.ceil((mid_p + 1) / mink)
        real_start_idx = index - after_padding_index // mink
        return padding, after_padding_index, real_start_idx


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True, use_small_conv=True):
    """
    一个分发函数，根据kernel_size的类型选择不同的卷积实现。
    """
    if isinstance(kernel_size, int) or len(set(kernel_size)) == 1:
        # 如果kernel_size是整数或所有维度相同（对称卷积核）
        # 则使用标准的conv_bn_ori构建块
        return conv_bn_ori(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            dilation,
            bn)
    else:
        # 如果kernel_size是元组且维度不同（非对称卷积核）
        # 则使用LoRAConvsByWeight来实现大核分解
        big_kernel, small_kernel = kernel_size
        return LoRAConvsByWeight(in_channels, out_channels, bn=bn,
                                 big_kernel=big_kernel, small_kernel=small_kernel,
                                 group=groups, stride=stride,
                                 use_small_conv=use_small_conv)


def fuse_bn(conv, bn):
    """
    将批归一化（BN）层的参数融合进它前面的卷积（Conv）层。
    这是一种常见的部署时优化，可以减少计算量。
    返回融合后的新权重和新偏置。
    """
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    # 计算融合后的权重和偏置
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):
    """
    大卷积核重参数化模块。
    在训练时，它可以使用多分支结构（例如，一个大核分解分支+一个小核分支）。
    在部署时，它可以将这些分支融合成一个单一的大卷积核，以加快推理速度。
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            small_kernel=5,
            stride=1,
            groups=1,
            small_kernel_merged=False, # 是否已经融合（用于部署）
            Decom=True, # 是否使用大核分解（LoRA）
            bn=True,
    ):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.Decom = Decom
        padding = kernel_size // 2
        
        if small_kernel_merged:  # 如果是部署模式（已融合）
            # 直接创建一个单一的、带有偏置的大卷积核
            self.lkb_reparam = get_conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=1, groups=groups, bias=True,
            )
        else: # 训练模式
            if self.Decom:
                # 使用大核分解（非对称卷积核）
                self.LoRA = conv_bn(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(kernel_size, small_kernel), stride=stride,
                    padding=padding, dilation=1, groups=groups, bn=bn
                )
            else:
                # 使用一个标准的大卷积核分支
                self.lkb_origin = conv_bn(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding,
                    dilation=1, groups=groups, bn=bn,
                )
            # 如果提供了小卷积核尺寸，则额外创建一个并行的小卷积核分支
            if (small_kernel is not None) and small_kernel < kernel_size:
                self.small_conv = conv_bn(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=small_kernel, stride=stride,
                    padding=small_kernel // 2, groups=groups, dilation=1, bn=bn,
                )
        
        # 最后的BN层和激活函数
        self.bn = get_bn(out_channels)
        self.act = nn.SiLU()

    def forward(self, inputs):
        # 根据模块的当前状态（训练或部署）选择不同的前向路径
        if hasattr(self, "lkb_reparam"):
            # 部署模式：直接使用融合后的大卷积核
            out = self.lkb_reparam(inputs)
        elif self.Decom:
            # 训练模式，且使用大核分解
            out = self.LoRA(inputs)
            if hasattr(self, "small_conv"):
                # 如果有小核分支，则将其结果相加
                out += self.small_conv(inputs)
        else:
            # 训练模式，使用标准大核
            out = self.lkb_origin(inputs)
            if hasattr(self, "small_conv"):
                # 如果有小核分支，则将其结果相加
                out += self.small_conv(inputs)
        # 最后的BN和激活
        return self.act(self.bn(out))

    def get_equivalent_kernel_bias(self):
        """计算多分支融合后的等效卷积核与偏置。"""
        # 首先融合主分支的Conv和BN
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            # 如果有小核分支，也融合其Conv和BN
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            # 偏置直接相加
            eq_b += small_b
            # 小核的权重需要填充到大核的中央再相加
            eq_k += nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )
        return eq_k, eq_b

    def switch_to_deploy(self):
        """将模块从训练模式切换到部署模式。"""
        if hasattr(self, 'lkb_origin'):
            # 计算融合后的等效权重和偏置
            eq_k, eq_b = self.get_equivalent_kernel_bias()
            # 创建一个新的、单一的卷积层用于部署
            self.lkb_reparam = get_conv2d(
                in_channels=self.lkb_origin.conv.in_channels,
                out_channels=self.lkb_origin.conv.out_channels,
                kernel_size=self.lkb_origin.conv.kernel_size,
                stride=self.lkb_origin.conv.stride,
                padding=self.lkb_origin.conv.padding,
                dilation=self.lkb_origin.conv.dilation,
                groups=self.lkb_origin.conv.groups,
                bias=True, # 融合后需要偏置
            )
            # 将计算好的等效权重和偏置赋值给新层
            self.lkb_reparam.weight.data = eq_k
            self.lkb_reparam.bias.data = eq_b
            # 删除训练时的分支，节省内存
            self.__delattr__("lkb_origin")
            if hasattr(self, "small_conv"):
                self.__delattr__("small_conv")