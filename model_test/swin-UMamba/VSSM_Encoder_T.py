import torch
import torch.nn as nn
import math
from Swin_UMamba_Models import *
# 假设 VSSMEncoder 已定义在此处...

# 示例输入

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = torch.randn(1, 3, 224, 224).to(device) # 批大小为1，3通道（如RGB），224x224的图像

# 实例化模型
model = VSSMEncoder(patch_size=2, in_chans=48)
model.to(device)
# 修改 forward 函数来打印每层的张量形状
class VSSMEncoderDebug(VSSMEncoder):
    def forward(self, x):
        x_ret = []
        x_ret.append(x)

        print(f"Input shape: {x.shape}")  # 打印输入张量的形状

        x = self.patch_embed(x)
        print(f"After Patch Embedding shape: {x.shape}")  # 打印补丁嵌入后的张量形状

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            x = layer(x)
            print(f"After Layer {s+1} shape: {x.shape}")  # 打印每层后的张量形状
            x_ret.append(x.permute(0, 3, 1, 2))
            if s < len(self.downsamples):
                x = self.downsamples[s](x)
                print(f"After Downsample {s+1} shape: {x.shape}")  # 打印下采样后的张量形状

        return x_ret

# 使用调试版的模型
debug_model = VSSMEncoderDebug().to(device)

# 前向传播并查看张量形状变化
with torch.no_grad():
    output = debug_model(input_tensor)
