import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision

''' Baseline encoder-decoder architecture
Activation  : Softmax
Loss        : Cross entropy
Optimizer   : Adam / AdamW

Use early stopping
'''
class FCN_baseline(nn.Module):

    def __init__(self, n_class=21, activation=nn.ReLU):
        super().__init__()
        self.n_class = n_class
        try:
            self.act = activation(inplace=True)
        except:
            self.act = activation()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5 = nn.BatchNorm2d(512)

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)


    def forward(self, x):
        
        x1 = self.bnd1(self.act(self.conv1(x)))
        x2 = self.bnd2(self.act(self.conv2(x1)))
        x3 = self.bnd3(self.act(self.conv3(x2)))
        x4 = self.bnd4(self.act(self.conv4(x3)))
        x5 = self.bnd5(self.act(self.conv5(x4)))
        
        y1 = self.bn1(self.act(self.deconv1(x5)))
        y2 = self.bn2(self.act(self.deconv2(y1)))
        y3 = self.bn3(self.act(self.deconv3(y2)))
        y4 = self.bn4(self.act(self.deconv4(y3)))
        y5 = self.bn5(self.act(self.deconv5(y4)))

        score = self.classifier(y5)

        return score  # (N, C, H, W)
        

class _DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class _Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class _Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = _DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class _OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=21, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = _DoubleConv(n_channels, 64)
        self.down1 = _Down(64, 128)
        self.down2 = _Down(128, 256)
        self.down3 = _Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = _Down(512, 1024 // factor)
        self.up1 = _Up(1024, 512 // factor, bilinear)
        self.up2 = _Up(512, 256 // factor, bilinear)
        self.up3 = _Up(256, 128 // factor, bilinear)
        self.up4 = _Up(128, 64, bilinear)
        self.outc = _OutConv(64, n_classes)
#         self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
#         logits = self.activation(self.outc(x)) #don't need softmax when using nn.CrossEntropyLoss
        logits = self.outc(x)
        return logits
    

class Resnet(nn.Module):

    def __init__(self, n_class=21):
        super().__init__()
        self.n_class = n_class
        self.relu = nn.ReLU(inplace=True)

        # Replace encoder of the given FCN architecture with ResNet34 for Transfer Learning
        rnet_org = torchvision.models.resnet34(pretrained=True)
        # Removed the last fully connected layer and the avgpool layer of ResNet34 to match the dimension of the decoder
        self.rnet = nn.Sequential(*(list(rnet_org.children())[:-2]))  

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        x1 = self.rnet(x)
        
        y1 = self.bn1(self.relu(self.deconv1(x1)))
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        y5 = self.bn5(self.relu(self.deconv5(y4)))

        score = self.classifier(y5)

        return score  # (N, C, H, W)
    
class New_arch(nn.Module):
    
    def __init__(self, n_class=21):
        super().__init__()
        self.n_class = n_class
        self.act = nn.Softmax()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd6 = nn.BatchNorm2d(1024)
        self.conv7 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd7 = nn.BatchNorm2d(2048)
        self.conv8 = nn.Conv2d(2048, 4096, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd8 = nn.BatchNorm2d(4096)
        
        self.deconv1 = nn.ConvTranspose2d(4096, 4096, kernel_size=3, stride=1, padding=1, dilation=1)#, output_padding=1)
        self.bn1 = nn.BatchNorm2d(4096)
        self.deconv2 = nn.ConvTranspose2d(4096, 2048, kernel_size=3, stride=1, padding=1, dilation=1)#, output_padding=1)
        self.bn2 = nn.BatchNorm2d(2048)
        self.deconv3 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=1, padding=1, dilation=1)#, output_padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.deconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.deconv6 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.deconv7 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.deconv8 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn8 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)


    def forward(self, x):
        
        x = self.bnd1(self.act(self.conv1(x)))
        x = self.bnd2(self.act(self.conv2(x)))
        x = self.bnd3(self.act(self.conv3(x)))
        x = self.bnd4(self.act(self.conv4(x)))
        x = self.bnd5(self.act(self.conv5(x)))
        x = self.bnd6(self.act(self.conv6(x)))
        x = self.bnd7(self.act(self.conv7(x)))
        x = self.bnd8(self.act(self.conv8(x)))
        
        y = self.bn1(self.act(self.deconv1(x)))
        y = self.bn2(self.act(self.deconv2(y)))
        y = self.bn3(self.act(self.deconv3(y)))
        y = self.bn4(self.act(self.deconv4(y)))
        y = self.bn5(self.act(self.deconv5(y)))
        y = self.bn6(self.act(self.deconv6(y)))
        y = self.bn7(self.act(self.deconv7(y)))
        y = self.bn8(self.act(self.deconv8(y)))

        score = self.classifier(y)

        return score  # (N, C, H, W)
