# -*- coding: utf-8 -*-
import jittor as jt
from jittor import init
from jittor import nn


'''--------------------------------------UNet--------------------------------------'''
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if (not mid_channels):
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv(in_channels, mid_channels, 3, padding=0),
            nn.BatchNorm(mid_channels),
            nn.ReLU(),
            nn.Conv(mid_channels, out_channels, 3, padding=0),
            nn.BatchNorm(out_channels),
            nn.ReLU()
        )

    def execute(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def execute(self, x):
        x = self.maxpool_conv(x)
        return self.dropout(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')  #使用双线性插值 通道数保持不变
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2) #通过卷积层改变通道
        else:
            self.up = nn.ConvTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)  #使用转置卷积上采样 通道数减半
            self.conv = DoubleConv(in_channels, out_channels)   # 这里输入通道为什么不是 in_channels // 2
                                                                # 因为两个 in_channels // 2 拼接后就变成了 in_channels

    def execute(self, x1, x2):
        #将x2 剪切为x1的形状 因为x1的尺寸比 x2小
        x1 = self.up(x1)

        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        crop_y1 = diffY // 2
        crop_y2 = crop_y1 + x1.shape[2]
        crop_x1 = diffX // 2
        crop_x2 = crop_x1 + x1.shape[3]

        x2_cropped = x2[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
        x = jt.contrib.concat([x2_cropped, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv(in_channels, out_channels, 1)

    def execute(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, bilinear=True, dropout_probs=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        if dropout_probs is None:
            dropout_probs = [0.0, 0.0, 0.0, 0.0]
        elif isinstance(dropout_probs, float):
            dropout_probs = [dropout_probs] * 4
        elif len(dropout_probs) < 4:
            dropout_probs = dropout_probs + [dropout_probs[-1]] * (4 - len(dropout_probs))

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, dropout_prob=dropout_probs[0])
        self.down2 = Down(128, 256, dropout_prob=dropout_probs[1])
        self.down3 = Down(256, 512, dropout_prob=dropout_probs[2])
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, dropout_prob=dropout_probs[3])

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # 初始化权重 参照论文
        self.initialize_weights()

                                #   maxpool不改变通道,改变形状  bilinear上采样不改变通道数  ConvTranspose上采样改变通道数
    def execute(self, x):       #通道变化    bilinear                                           nobilinear
        x1 = self.inc(x)              #     1->64(中间层通道数)->64                               1->64->64
        x2 = self.down1(x1)           #     64->128->128                                        64->128->128
        x3 = self.down2(x2)           #     128->256->256                                       128->256->256
        x4 = self.down3(x3)           #     256->512->512                                       256->512->512
        x5 = self.down4(x4)           #     512->512->512                                       512->1024->1024
        x = self.up1(x5, x4)    #cat(upsample(512->512),512)= 1024->512->256  c((1024->512),512)=1024->512->512
        x = self.up2(x, x3)     #  c((256->256),256)=512->256->128                c((512->256),256)=512->256->256
        x = self.up3(x, x2)     #  c((128->128),128)=256->128->64                 c((256->128),128)=256->128->128
        x = self.up4(x, x1)     #  c((64->64),64)=128->64->64                     c((128->64),64)=128->64->64
        logits = self.outc(x)         #  64->2                                                  64->2
        return logits

    def initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv, nn.ConvTranspose)):
                # 计算传入连接数 N
                kernel_size = module.kernel_size
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)
                in_channels = module.in_channels
                N = in_channels * kernel_size[0] * kernel_size[1]

                # 计算标准差 std = sqrt(2/N)
                std = (2.0 / N) ** 0.5

                # 从高斯分布初始化权重
                init.gauss_(module.weight, 0.0, std)

                # 初始化偏置为零
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)

            elif isinstance(module, nn.BatchNorm):
                # BatchNorm 层初始化
                init.constant_(module.weight, 1.0)
                init.constant_(module.bias, 0.0)

    def get_loss(self, pred, target, ignore_index=None):
        loss = nn.cross_entropy_loss(pred, target, ignore_index=ignore_index)
        return loss

    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()






def main_Unet():
    jt.flags.use_cuda = 1
    model = UNet(n_channels=3,bilinear=True, dropout_probs=[0.1, 0.2, 0.3, 0.4])
    x = jt.ones([2, 3, 448, 448])  # B N_channel w h
    y = model(x)
    print(y.shape)
    _ = y.data

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    '''
    UNet
    17,266,306 total parameters.
    17,266,306 training parameters.
    Input size (MB): 2.30
    Forward/backward pass size (MB): 1990.86
    Params size (MB): 65.87
    Estimated Total Size (MB): 2059.03
    '''
    print('-----------------------jittorsummary----------------------------')
    #利用jittorsummary统计更详细的参数
    from jittorsummary import summary
    summary(model, input_size=(3, 448, 448))




'''--------------------------------------U2Net--------------------------------------'''

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        # Conv2D + BatchNorm + ReLU 的基础组合
        # dilation为1相当于普通卷积
        # padding=(1 * dirate) 保持尺寸不变
        # 原因: 对于stride=1 如果尺寸不变 要求 padding=(dilation*(kernel_size-1))/2
        self.conv_s1 = nn.Conv(in_ch, out_ch, 3, padding=(1 * dirate), dilation=(1 * dirate))
        self.bn_s1 = nn.BatchNorm(out_ch)
        self.relu_s1 = nn.ReLU()

    def execute(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout


def _upsample_like(src, tar):
    # 将源张量 src 上采样至目标张量 tar 的空间尺寸（高度和宽度），使用双线性插值
    # B*C*H*W
    src = nn.upsample(src, size=tar.shape[2:], mode='bilinear')
    return src


# RSU-7 所有的RSU都是根据论文中的RSU结构 和 论文中的参数搭建的 可以看PNG文件夹
# RSU-L L是编码器层数
#   输出尺寸分辨率不变
class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum') #ceil_mode=True 使用向上取整，保留边界信息
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV((mid_ch * 2), out_ch, dirate=1)

    def execute(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(jt.contrib.concat((hx7, hx6), dim=1))
        hx6dup = _upsample_like(hx6d, hx5)
        hx5d = self.rebnconv5d(jt.contrib.concat((hx6dup, hx5), dim=1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(jt.contrib.concat((hx5dup, hx4), dim=1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(jt.contrib.concat((hx4dup, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(jt.contrib.concat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(jt.contrib.concat((hx2dup, hx1), dim=1))
        return (hx1d + hxin)


# RSU-6 与RSU-7同理 输出尺寸分辨率不变
class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv5d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV((mid_ch * 2), out_ch, dirate=1)

    def execute(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(jt.contrib.concat((hx6, hx5), dim=1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(jt.contrib.concat((hx5dup, hx4), dim=1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(jt.contrib.concat((hx4dup, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(jt.contrib.concat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(jt.contrib.concat((hx2dup, hx1), dim=1))
        return (hx1d + hxin)


# RSU-5 与RSU-7同理 输出尺寸分辨率不变
class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV((mid_ch * 2), out_ch, dirate=1)

    def execute(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(jt.contrib.concat((hx5, hx4), dim=1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(jt.contrib.concat((hx4dup, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(jt.contrib.concat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(jt.contrib.concat((hx2dup, hx1), dim=1))
        return (hx1d + hxin)


# RSU-4 与RSU-7同理 输出尺寸分辨率不变
class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV((mid_ch * 2), mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV((mid_ch * 2), out_ch, dirate=1)

    def execute(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(jt.contrib.concat((hx4, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(jt.contrib.concat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(jt.contrib.concat((hx2dup, hx1), dim=1))
        return (hx1d + hxin)


# RSU-4F No maxpooling  输出尺寸分辨率不变
#RSU-4F 的 dirate可以从U2net模型图的图例中找到
class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV((mid_ch * 2), mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV((mid_ch * 2), mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV((mid_ch * 2), out_ch, dirate=1)

    def execute(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(jt.contrib.concat((hx4, hx3), dim=1))
        hx2d = self.rebnconv2d(jt.contrib.concat((hx3d, hx2), dim=1))
        hx1d = self.rebnconv1d(jt.contrib.concat((hx2d, hx1), dim=1))
        return (hx1d + hxin)


##### U2-Net #####
class U2net(nn.Module):
    def __init__(self, in_ch=3, n_classes=2):
        super(U2net, self).__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        # encoder
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage6 = RSU4F(512, 256, 512)
        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        #论文中提到的 通过3*3卷积和Sigmod函数 生成 显著概率图     simoid函数在execute中能看到(最后被注释掉的部分)
        self.side1 = nn.Conv(64, n_classes, 3, padding=1)
        self.side2 = nn.Conv(64, n_classes, 3, padding=1)
        self.side3 = nn.Conv(128, n_classes, 3, padding=1)
        self.side4 = nn.Conv(256, n_classes, 3, padding=1)
        self.side5 = nn.Conv(512, n_classes, 3, padding=1)
        self.side6 = nn.Conv(512, n_classes, 3, padding=1)
        # 论文中提到的 将输出的显著图的逻辑图(卷积输出，Sigmoid函数之前) 向上采样至与输入图像大小一致 然后通过1*1卷积和一个simoid函数生成最终的Sfuse
        # 逻辑图的上采样操作看execute
        self.outconv = nn.Conv((6 * n_classes), n_classes, 1)

    '''
    ---尺寸变化---
    经过RSU的尺寸是不会变的
    但经过RSU之后,下采样或上采样的尺寸是会变的
    '''
    def execute(self, x):
        # -------------------- encoder -------------------
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder -------------------
        hx5d = self.stage5d(jt.contrib.concat((hx6up, hx5), dim=1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.stage4d(jt.contrib.concat((hx5dup, hx4), dim=1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.stage3d(jt.contrib.concat((hx4dup, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(jt.contrib.concat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(jt.contrib.concat((hx2dup, hx1), dim=1))

        # side output
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)
        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)
        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)
        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)
        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)
        # 如果需要监督每一个side的输出就要返回d0到d6,我这里就是在Unet论文的框架下换一个model,所以没有复现U2Net的损失函数,用的是Unet的损失函数
        # 所以返回d0作为最后输出结果 d0不需要经过Sigmod,因为损失函数中会有softmax
        d0 = self.outconv(jt.contrib.concat((d1, d2, d3, d4, d5, d6), dim=1))
        # return (jt.sigmoid(d0), jt.sigmoid(d1), jt.sigmoid(d2), jt.sigmoid(d3), jt.sigmoid(d4), jt.sigmoid(d5), jt.sigmoid(d6))
        return d0

    def get_loss(self, target, d0, d1, d2, d3, d4, d5, d6, ignore_index=None):
        loss0 = nn.cross_entropy_loss(d0, target, ignore_index=ignore_index)  # tar loss
        loss1 = nn.cross_entropy_loss(d1, target, ignore_index=ignore_index)
        loss2 = nn.cross_entropy_loss(d2, target, ignore_index=ignore_index)
        loss3 = nn.cross_entropy_loss(d3, target, ignore_index=ignore_index)
        loss4 = nn.cross_entropy_loss(d4, target, ignore_index=ignore_index)
        loss5 = nn.cross_entropy_loss(d5, target, ignore_index=ignore_index)
        loss6 = nn.cross_entropy_loss(d6, target, ignore_index=ignore_index)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
        loss5.data.item(), loss6.data.item()))

        return loss0, loss

    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


### U2-Net small ###
class U2netS(nn.Module):
    def __init__(self, in_ch=3, n_classes=2):
        super(U2netS, self).__init__()
        self.in_ch = in_ch
        self.n_classes=n_classes
        # encoder
        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv(64, n_classes, 3, padding=1)
        self.side2 = nn.Conv(64, n_classes, 3, padding=1)
        self.side3 = nn.Conv(64, n_classes, 3, padding=1)
        self.side4 = nn.Conv(64, n_classes, 3, padding=1)
        self.side5 = nn.Conv(64, n_classes, 3, padding=1)
        self.side6 = nn.Conv(64, n_classes, 3, padding=1)
        self.outconv = nn.Conv((6 * n_classes), n_classes, 1)

    def execute(self, x):
        # -------------------- encoder -------------------
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder -------------------
        hx5d = self.stage5d(jt.contrib.concat((hx6up, hx5), dim=1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.stage4d(jt.contrib.concat((hx5dup, hx4), dim=1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.stage3d(jt.contrib.concat((hx4dup, hx3), dim=1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(jt.contrib.concat((hx3dup, hx2), dim=1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(jt.contrib.concat((hx2dup, hx1), dim=1))

        # side output
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)
        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)
        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)
        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)
        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)
        d0 = self.outconv(jt.contrib.concat((d1, d2, d3, d4, d5, d6), dim=1))
        # 如果需要监督每一个side的输出就要返回d0到d6,我这里就是在Unet论文的框架下换一个model,所以没有复现U2Net的损失函数,用的是Unet的损失函数
        # 所以返回d0作为最后输出结果 d0不需要经过Sigmod,因为损失函数中会有softmax
        return d0
        # return (jt.sigmoid(d0), jt.sigmoid(d1), jt.sigmoid(d2), jt.sigmoid(d3), jt.sigmoid(d4), jt.sigmoid(d5), jt.sigmoid(d6))

    def get_loss(self, target, d0, d1, d2, d3, d4, d5, d6, ignore_index=None):
        loss0 = nn.cross_entropy_loss(d0, target, ignore_index=ignore_index)  # tar loss
        loss1 = nn.cross_entropy_loss(d1, target, ignore_index=ignore_index)
        loss2 = nn.cross_entropy_loss(d2, target, ignore_index=ignore_index)
        loss3 = nn.cross_entropy_loss(d3, target, ignore_index=ignore_index)
        loss4 = nn.cross_entropy_loss(d4, target, ignore_index=ignore_index)
        loss5 = nn.cross_entropy_loss(d5, target, ignore_index=ignore_index)
        loss6 = nn.cross_entropy_loss(d6, target, ignore_index=ignore_index)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
        loss5.data.item(), loss6.data.item()))

        return loss0, loss

    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main_U2net():
    jt.flags.use_cuda = 1
    model = U2netS()
    x = jt.ones([2, 3, 260, 260])  # B N_channel w h
    y = model(x)
    print(y.shape)
    _ = y.data

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    '''
    U2Net
    44,052,518 total parameters.
    44,023,718 training parameters.
    260*260:
        Input size (MB): 0.77
        Forward/backward pass size (MB): 1336.23
        Params size (MB): 167.94
        Estimated Total Size (MB): 1504.94
    448*448:
            Input size (MB): 2.30
        Forward/backward pass size (MB): 3938.61
        Params size (MB): 167.94
        Estimated Total Size (MB): 4108.84
    '''


    '''
    U2NetS 
    1,140,358 total parameters.
    1,134,662 training parameters.
    260*260:
        Input size (MB): 0.77
        Forward/backward pass size (MB): 962.83
        Params size (MB): 4.33
        Estimated Total Size (MB): 967.93
    448*448:
        Input size (MB): 2.30
        Forward/backward pass size (MB): 2854.31
        Params size (MB): 4.33
        Estimated Total Size (MB): 2860.94
    '''
    print('-----------------------jittorsummary----------------------------')
    #利用jittorsummary统计更详细的参数
    from jittorsummary import summary
    summary(model, input_size=(3, 260, 260))


if __name__ == '__main__':
    #main_Unet()
    main_U2net()