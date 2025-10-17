import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules import Conv
from ..extra_modules.transformer import DynamicTanh
__all__ = ['PPA', 'DASI']

# --- 空间注意力模块 (Spatial Attention Module, SAM) ---
# 该模块计算空间注意力图，以根据特征的空间位置来增强或抑制它们。
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        """
        初始化空间注意力模块。
        它使用一个卷积层从聚合的通道信息中创建一个注意力掩码。
        """
        super(SpatialAttentionModule, self).__init__()
        # 一个二维卷积层，接收拼接后的平均池化和最大池化特征（2个通道），
        # 并输出一个单通道的注意力图。
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # Sigmoid激活函数，将注意力图的值缩放到0和1之间。
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        空间注意力模块的前向传播。
        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)。
        Returns:
            torch.Tensor: 应用了空间注意力的特征图。
        """
        # 沿通道维度进行平均池化。keepdim=True保持维度为 (B, 1, H, W)。
        avgout = torch.mean(x, dim=1, keepdim=True)
        # 沿通道维度进行最大池化。keepdim=True保持维度为 (B, 1, H, W)。
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        
        # 沿通道维度拼接平均池化和最大池化的特征，形状变为 (B, 2, H, W)。
        out = torch.cat([avgout, maxout], dim=1)
        # 应用卷积生成单通道的空间注意力图，
        # 然后通过Sigmoid函数获得范围在0到1之间的权重。
        out = self.sigmoid(self.conv2d(out))
        
        # 将原始输入特征图与空间注意力图逐元素相乘。
        # 这会根据每个空间位置的重要性来缩放特征。
        return out * x

# --- 局部-全局注意力模块 (Local-Global Attention) ---
# 该模块在局部图块中处理特征，并使用可学习的“提示”(prompt)来整合全局上下文。
class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        """
        初始化LocalGlobalAttention模块。
        Args:
            output_dim (int): 输出通道数/特征维度。
            patch_size (int): 用于局部处理的正方形图块的大小。
        """
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        
        # 用于处理展平后图块特征的多层感知机 (MLP)。
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        
        # 用于优化最终输出的1x1卷积。
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        
        # 一个可学习的参数，代表一个全局的“提示”或查询向量。
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True)) 
        # 一个可学习的变换矩阵，应用于注意力加权后的特征。
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        """
        LocalGlobalAttention模块的前向传播。
        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)。
        Returns:
            torch.Tensor: 处理后的特征图。
        """
        # 将张量形状从 (B, C, H, W) 变为 (B, H, W, C) 以方便处理图块。
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # --- 局部分支 ---
        # 从输入特征图中创建不重叠的 PxP 大小的图块。
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # 形状: (B, H/P, W/P, C, P, P)
        # 重塑形状以组合图块。形状: (B, num_patches, C, P*P)。
        local_patches = local_patches.reshape(B, -1, C, P*P)
        # 在每个图块内对特征进行平均。形状: (B, num_patches, P*P)
        # 注意: 这里是沿通道维度 `C` 进行平均，这是一种不寻常的操作。
        # 更常见的方法是展平 C, H, W 或在空间维度上平均。
        # 此处我们遵循原始代码的写法。
        local_patches = local_patches.mean(dim=2)

        # 通过MLP处理图块特征。
        local_patches = self.mlp1(local_patches)
        local_patches = self.norm(local_patches)
        local_patches = self.mlp2(local_patches)

        # 为每个图块特征向量计算注意力分数并应用它。
        local_attention = F.softmax(local_patches, dim=-1)
        local_out = local_patches * local_attention

        # --- 使用Prompt进行全局交互 ---
        # 计算每个图块输出与全局可学习提示之间的余弦相似度。
        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1) # 形状: (B, num_patches, 1)
        # 将相似度限制在[0, 1]之间以创建一个软掩码。
        mask = cos_sim.clamp(0, 1)
        # 应用掩码来过滤或加权局部图块的输出。
        local_out = local_out * mask
        # 应用另一个可学习的线性变换。
        local_out = local_out @ self.top_down_transform

        # --- 恢复形状 ---
        # 将处理后的图块向量重塑回空间网格形状。
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim) # 形状: (B, H/P, W/P, output_dim)
        # 变换回 (B, C, H, W) 格式。
        local_out = local_out.permute(0, 3, 1, 2)
        # 使用双线性插值将特征图上采样至原始输入分辨率。
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        # 使用1x1卷积进行最终的优化。
        output = self.conv(local_out)

        return output

# --- 高效通道注意力 (Efficient Channel Attention, ECA) ---
# 该模块自适应地计算通道注意力，并根据通道维度动态确定卷积核的大小。
class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        """
        初始化ECA模块。
        Args:
            in_channel (int): 输入通道数。
            gamma (int): 用于控制卷积核大小自适应的参数。
            b (int): 用于控制卷积核大小自适应的参数。
        """
        super(ECA, self).__init__()
        # 根据通道数自适应地计算一维卷积的卷积核大小。
        # 这样可以避免手动调整卷积核大小。
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1 # 卷积核大小必须是奇数。
        padding=kernel_size//2
        
        # 全局平均池化，将空间信息聚合成一个通道描述符。
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        # 一维卷积用于捕获局部跨通道的交互信息，然后通过Sigmoid获得注意力权重。
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        """
        ECA模块的前向传播。
        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)。
        Returns:
            torch.Tensor: 应用了通道注意力的特征图。
        """
        # 应用全局平均池化。形状: (B, C, 1, 1)。
        out=self.pool(x)
        # 为一维卷积重塑形状：(B, C, 1, 1) -> (B, 1, C)。
        out=out.view(x.size(0),1,x.size(1))
        # 应用一维卷积和Sigmoid函数来获得通道权重。
        out=self.conv(out)
        # 重塑回 (B, C, 1, 1) 的形状，以便与输入进行广播（element-wise multiplication）。
        out=out.view(x.size(0),x.size(1),1,1)
        # 将原始输入与通道注意力权重相乘。
        return out*x

# --- 渐进式金字塔注意力模块 (Progressive Pyramid Attention, PPA) ---
# 该模块聚合了来自多个卷积路径和注意力机制的特征。
class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
        """
        初始化PPA模块。
        Args:
            in_features (int): 输入通道数。
            filters (int): 内部和输出的通道数。
        """
        super().__init__()
        # 假设'Conv'是一个预定义的'卷积-BN-激活'块。为简单起见，这里使用nn.Conv2d。
        # 跳跃连接路径。
        self.skip = nn.Conv2d(in_features, filters, kernel_size=1)
        # 三个顺序连接的卷积层。
        self.c1 = nn.Conv2d(filters, filters, 3, padding=1)
        self.c2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.c3 = nn.Conv2d(filters, filters, 3, padding=1)
        
        # 注意力模块。
        self.sa = SpatialAttentionModule() # 空间注意力
        self.cn = ECA(filters) # 通道注意力
        self.lga2 = LocalGlobalAttention(filters, 2) # 使用 2x2 图块的局部-全局注意力
        self.lga4 = LocalGlobalAttention(filters, 4) # 使用 4x4 图块的局部-全局注意力

        self.drop = nn.Dropout2d(0.1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.silu = nn.SiLU()

    def forward(self, x):
        """
        PPA模块的前向传播。
        Args:
            x (torch.Tensor): 输入特征图。
        Returns:
            torch.Tensor: 处理后的特征图。
        """
        # 一个用于聚合的独立跳跃连接路径。
        x_skip = self.skip(x)
        # 两个并行的、具有不同图块大小的Local-Global Attention路径。
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        
        # 一个包含三个顺序卷积的路径。
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        
        # 通过逐元素相加来聚合所有路径的特征。
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        
        # 依次应用通道注意力和空间注意力。
        x = self.cn(x)
        x = self.sa(x)
        
        # 最后的处理步骤。
        x = self.drop(x)
        x = self.bn1(x)
        x = self.silu(x)
        return x

# --- 用于特征门控的Bag模块 ---
# 一个简单的模块，使用第三个特征图(d)作为门控，来混合两个特征图(p, i)。
class Bag(nn.Module):
    def __init__(self):
        super(Bag, self).__init__()
    def forward(self, p, i, d):
        """
        Bag模块的前向传播。
        Args:
            p (torch.Tensor): 主要特征图。
            i (torch.Tensor): 次要特征图。
            d (torch.Tensor): 门控特征图 (例如，边缘注意力)。
        Returns:
            torch.Tensor: 混合后的特征图。
        """
        # 从门控图 `d` 计算注意力权重。
        edge_att = torch.sigmoid(d)
        # 对两个输入特征进行加权求和。
        # 如果 edge_att 值高，`p` 的权重更大；如果值低，`i` 的权重更大。
        return edge_att * p + (1 - edge_att) * i

# --- 双分支聚合与尺度交互模块 (Dual-branch Aggregation and Scale Interaction, DASI) ---
# 一个融合模块，旨在结合来自三个不同尺度（低、中、高分辨率）的特征。
class DASI(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        """
        初始化DASI模块。
        Args:
            in_features (list[int]): 一个包含[低, 中, 高]分辨率特征输入通道数的列表。
            out_features (int): 输出通道数。
        """
        super().__init__()
        self.bag = Bag()
        self.tail_conv = nn.Conv2d(out_features, out_features, 1)
        self.conv = nn.Conv2d(out_features // 2, out_features // 4, 1)
        self.bns = nn.BatchNorm2d(out_features)

        # 用于处理和对齐来自不同尺度特征的通道的卷积层。
        self.skips = nn.Conv2d(in_features[1], out_features, 1) # 用于中分辨率特征
        self.skips_2 = nn.Conv2d(in_features[0], out_features, 1) # 用于低分辨率特征
        # 对于高分辨率特征，使用带步幅和空洞的卷积进行下采样。
        self.skips_3 = nn.Conv2d(in_features[2], out_features, kernel_size=3, stride=2, dilation=2, padding=2)
        
        self.silu = nn.SiLU()

    def forward(self, x_list):
        """
        DASI模块的前向传播。
        Args:
            x_list (list[torch.Tensor]): 一个包含三个特征图的列表 [低分辨率, 中分辨率, 高分辨率]。
                                         对于网络的最高层或最底层，可以使用 `None`。
        Returns:
            torch.Tensor: 融合后的特征图。
        """
        x_low, x, x_high = x_list # 解包特征：低分辨率、中分辨率、高分辨率

        # --- 处理来自不同尺度的特征以对齐它们 ---
        if x_high is not None:
            x_high = self.skips_3(x_high) # 下采样高分辨率特征
            x_high = torch.chunk(x_high, 4, dim=1) # 沿通道维度分割成4部分
        
        if x_low is not None:
            x_low = self.skips_2(x_low) # 处理低分辨率特征
            # 上采样以匹配中分辨率特征 `x` 的空间尺寸。
            x_low = F.interpolate(x_low, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)
            x_low = torch.chunk(x_low, 4, dim=1) # 分割成4部分
        
        x = self.skips(x) # 处理中分辨率特征
        x_skip = x # 保存用于残差连接
        x = torch.chunk(x, 4, dim=1) # 分割成4部分

        # --- 根据可用尺度融合特征 ---
        if x_high is None: # 网络的顶层（只有低分辨率和中分辨率可用）
            x0 = self.conv(torch.cat((x[0], x_low[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_low[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_low[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_low[3]), dim=1))
        elif x_low is None: # 网络的底层（只有中分辨率和高分辨率可用）
            x0 = self.conv(torch.cat((x[0], x_high[0]), dim=1))
            x1 = self.conv(torch.cat((x[0], x_high[1]), dim=1)) # 注意: 可能是个bug？这里可能应该是 x[1]
            x2 = self.conv(torch.cat((x[0], x_high[2]), dim=1)) # 注意: 可能是个bug？这里可能应该是 x[2]
            x3 = self.conv(torch.cat((x[0], x_high[3]), dim=1)) # 注意: 可能是个bug？这里可能应该是 x[3]
        else: # 网络的中间层（三个尺度都可用）
            # 使用Bag模块来混合低分辨率和高分辨率特征，并由中分辨率特征进行门控。
            x0 = self.bag(x_low[0], x_high[0], x[0])
            x1 = self.bag(x_low[1], x_high[1], x[1])
            x2 = self.bag(x_low[2], x_high[2], x[2])
            x3 = self.bag(x_low[3], x_high[3], x[3])

        # --- 最后的聚合与输出 ---
        # 将处理后的块重新拼接在一起。
        x = torch.cat((x0, x1, x2, x3), dim=1)
        # 应用一个最终的1x1卷积。
        x = self.tail_conv(x)
        # 添加残差连接。
        x += x_skip
        # 应用批归一化和激活函数。
        x = self.bns(x)
        x = self.silu(x)

        return x