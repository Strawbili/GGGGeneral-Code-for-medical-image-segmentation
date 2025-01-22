import torch
import torch.nn as nn

# 定义 PatchEmbed2D 类（请确保你的代码在这个脚本中可用或已经正确导入）
class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建一个输入张量
input_tensor = torch.randn(1, 3, 224, 224).to(device)  # 1张图片，3个通道（RGB），224x224尺寸

# 实例化 PatchEmbed2D 类并将其移动到GPU
model = PatchEmbed2D(patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm).to(device)

# 前向传播并打印每个步骤的输出形状
with torch.no_grad():  # 禁用梯度计算以提高效率
    print(f"Input shape: {input_tensor.shape}")  # Input shape: torch.Size([1, 3, 224, 224])
    output = model(input_tensor)
    print(f"Output shape after Patch Embedding: {output.shape}")  # Output shape after Patch Embedding: torch.Size([1, 56, 56, 96])
