import torch
import torch.nn as nn
from Swin_UMamba_Models import *

# 假设 `SwinUMamba` 模型类已经定义

def test_model():
    # 创建模型实例
    model = SwinUMamba_T(
        in_chans=1,
        out_chans=13,
        feat_size=[48, 96, 192, 384, 768],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size=768,
        norm_name="instance",
        res_block=True,
        spatial_dims=2,
        deep_supervision=False
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 创建一个输入张量，假设输入是 1 通道的 224x224 图像
    input_tensor = torch.randn(1, 1, 224, 224)  # Batch size = 1, Channels = 1, Height = 224, Width = 224
    input_tensor = input_tensor.to(device)
    print("Input Tensor Shape:", input_tensor.shape)

    # 前向传播，并打印每一层的输出维度
    with torch.no_grad():
        x1 = model.stem(input_tensor)
        print("Stem Output Shape:", x1.shape)

        vss_outs = model.vssm_encoder(x1)
        for i, vss_out in enumerate(vss_outs):
            print(f"vss_outs [{i}] Output Shape: {vss_out.shape}")

        enc1 = model.encoder1(input_tensor)
        print("Encoder 1 Output Shape:", enc1.shape)

        enc2 = model.encoder2(vss_outs[0])
        print("Encoder 2 Output Shape:", enc2.shape)

        enc3 = model.encoder3(vss_outs[1])
        print("Encoder 3 Output Shape:", enc3.shape)

        enc4 = model.encoder4(vss_outs[2])
        print("Encoder 4 Output Shape:", enc4.shape)

        enc5 = model.encoder5(vss_outs[3])
        print("Encoder 5 Output Shape:", enc5.shape)

        enc_hidden = vss_outs[4]
        print("Encoder Hidden Output Shape:", enc_hidden.shape)

        dec4 = model.decoder6(enc_hidden, enc5)
        print("Dec4 Output Shape:", dec4.shape)

        dec3 = model.decoder5(dec4, enc4)
        print("Dec3 Output Shape:", dec3.shape)

        dec2 = model.decoder4(dec3, enc3)
        print("Dec2 Output Shape:", dec2.shape)

        dec1 = model.decoder3(dec2, enc2)
        print("Dec1 Output Shape:", dec1.shape)

        dec0 = model.decoder2(dec1, enc1)
        print("Dec0 Output Shape:", dec0.shape)

        dec_out = model.decoder1(dec0)
        print("Dec_out Output Shape:", dec_out.shape)

        # 打印最后的输出形状
        if model.deep_supervision:
            feat_out = [dec_out, dec1, dec2, dec3]
            out = []
            for i in range(4):
                pred = model.out_layers[i](feat_out[i])
                out.append(pred)
                print(f"Output Layer {i} Shape: {pred.shape}")
        else:
            out = model.out_layers[0](dec_out)
            print("Final Output Shape:", out.shape)

if __name__ == "__main__":
    test_model()
