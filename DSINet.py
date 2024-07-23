import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from lib.models.cls_hrnet import HighResolutionNet
from lib.models.swin import SwinTransformer
from lib.models.capsules import PWV
# state_dict_path = './backboneweight/hrnetv2_w48_imagenet_pretrained.pth'
# with open(
#         r"./lib/config/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
#         'r', encoding='utf-8') as f:
#     temp = yaml.load(stream=f, Loader=yaml.FullLoader)
# hrnet = HighResolutionNet(temp)
# hrnet.load_state_dict(torch.load(state_dict_path))

swin_path='./backboneweight/swin_base_patch4_window12_384_22k.pth'
swin=SwinTransformer(img_size=384,embed_dim=128,depths=[2,2,18,2],num_heads=[4,8,16,32], window_size=384//32)
swin.load_state_dict(torch.load(swin_path, map_location='cpu')['model'], strict=False)

class ConvBnRelu(nn.Module):
    """Conv+BN+Relu layer"""
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class MCE(nn.Module):
    """Multi-scale Context Extraction Module (MCE)"""
    def __init__(self, in_channels, out_channels):
        super(MCE, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.ca = CA(in_channels * 2)
    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)
    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)
    def forward(self, x):
        size = x.size()[2:]
        B, C, H, W = x.size()
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat2_ = feat1 * feat2
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat3_ = feat2 * feat3
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        feat4_ = feat3 * feat4
        X = torch.cat([x, feat1, feat2_, feat3_, feat4_], dim=1)  # concat 四个池化的结果 2N个channel
        X = self.ca(X)
        a = torch.split(X, [C//4, C//4, C//4, C//4, C], dim=1)  # 分割通道(n/4, n/4, n/4, n/4, N)
        a1 = a[0] * feat1
        a2 = a[1] * feat2
        a3 = a[2] * feat3
        a4 = a[3] * feat4
        a5 = a[4] * x
        M = torch.cat([a1, a2, a3, a4, a5], dim=1)
        x = self.out(M)
        return x

class AFE(nn.Module):
    """Attention Feature Enhancement Module (AFE)"""
    def __init__(self):
        super(AFE, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.cbr1 = ConvBnRelu(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.cbr2 = ConvBnRelu(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = 128
        self.reduction = 4
        self.mlp = nn.Sequential(
            nn.Conv2d(self.channel, self.channel // self.reduction, 1, bias=False),  # Conv2d比Linear方便操作
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel // self.reduction, self.channel, 1, bias=False))

        # spatial attention
        self.sigmoid = nn.Sigmoid()
        self.cbr3 = ConvBnRelu(in_channels=128, out_channels=64, kernel_size=(1, 9), padding=(0, 4))
        self.cbr4 = ConvBnRelu(in_channels=64, out_channels=1, kernel_size=(9, 1), padding=(4, 0))
        self.cbr5 = ConvBnRelu(in_channels=128, out_channels=64, kernel_size=(9, 1), padding=(4, 0))
        self.cbr6 = ConvBnRelu(in_channels=64, out_channels=1, kernel_size=(1, 9), padding=(0, 4))

        self.cbr7 = ConvBnRelu(in_channels=256, out_channels=128, kernel_size=3, padding=1)  # XY= torch.cat([X1, Y], dim=1)
        self.downsample1 = ConvBnRelu(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.downsample2 = ConvBnRelu(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1))
    def forward(self, fea0, fea1, fea2, fea3):
        outx = self.cbr1(torch.cat([self.up(fea3), fea2], 1))
        outy = self.cbr2(torch.cat([self.up(fea1), fea0], 1))
        avg_out = self.mlp(self.avg_pool(outx))
        channel_out = self.sigmoid(avg_out)
        X = channel_out * outx + outx
        X1 = self.up(X)
        X1 = self.up(X1)

        sa1 = self.cbr4(self.cbr3(outy))
        sa2 = self.cbr6(self.cbr5(outy))
        sa = sa1 + sa2
        spatial_out = self.sigmoid(sa)
        Y = spatial_out * outy + outy
        XY = torch.concat([X1, Y], dim=1)
        XY = self.cbr7(XY)

        out1 = XY
        out2 = self.downsample1(out1)
        out3 = self.downsample2(out2)
        out4 = self.downsample3(out3)
        return out1, out2, out3, out4

class SIM(nn.Module):
    """Semantic Interaction Module (SIM)"""
    def __init__(self, inc, outc):
        super(SIM, self).__init__()
        self.cs1 = ConvBnRelu(inc, outc, kernel_size=3, stride=1, padding=1)
        self.cs2 = ConvBnRelu(inc, outc, kernel_size=3, stride=1, padding=1)
        self.cs3 = ConvBnRelu(inc, outc, kernel_size=3, stride=1, padding=1)
        self.cs4 = ConvBnRelu(inc, outc, kernel_size=3, stride=1, padding=1)

        self.ss1 = ConvBnRelu(inc, outc, kernel_size=3, stride=1, padding=1)
        self.ss2 = ConvBnRelu(inc, outc, kernel_size=3, stride=1, padding=1)
        self.ss3 = ConvBnRelu(inc, outc, kernel_size=3, stride=1, padding=1)
        self.ss4 = ConvBnRelu(inc, outc, kernel_size=3, stride=1, padding=1)

    def forward(self, cs, ss):
        if ss.size()[2:] != cs.size()[2:]:
            ss = F.interpolate(ss, size=cs.size()[2:], mode='bilinear')
        cs1 = self.cs1(cs )
        cs2 = self.cs2(cs1)
        ss1 = self.ss1(ss )
        ss2 = self.ss2(ss1)
        wr  = cs2*ss2
        cs3 = self.cs3(wr )+cs1
        cs4 = self.cs4(cs3)
        ss3 = self.ss3(wr )+ss1
        ss4 = self.ss4(ss3)
        return cs4, ss4
class ER(nn.Module):
    """ Edge Refinement stage"""
    def __init__(self, inc, outc):
        super(ER, self).__init__()
        self.conv_ext = nn.Sequential(nn.Conv2d(inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))
        self.conv_fe = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.conv_E = nn.Sequential(nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))
        self.conv_Fe = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
    def tensor_erode(self, bin_img, ksize=3):
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
        return eroded
    def tensor_dilate(self, bin_img, ksize=3):  #
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        dilate = patches.reshape(B, C, H, W, -1)
        dilate, _ = dilate.max(dim=-1)
        return dilate

    def forward(self, x):
        x = self.conv_ext(x)
        fe = self.conv_fe(x)
        d = self.tensor_dilate(fe)
        e = self.tensor_erode(fe)
        Ed = d - e
        E = self.conv_E(Ed)
        fea = x * (1 + E)
        Fe = self.conv_Fe(fea)
        return Fe
class ERF(nn.Module):
    """Edge Refinement Fusion Module (ERF)"""
    def __init__(self, k_size=3):
        super(ERF, self).__init__()
        self.er = ER(128,128)
        self.cbr1 = ConvBnRelu(in_channels=129, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cbr2 = ConvBnRelu(in_channels=129, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, cs, ss, Ci):
        Cie = self.er(Ci)
        cs = self.cbr1(torch.concat([cs, Cie], 1))
        ss = self.cbr2(torch.concat([ss, Cie], 1))
        y = self.avg_pool(cs)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return ss * y.expand_as(ss)
class CA(nn.Module):
    """channel attention"""
    def __init__(self, in_planes, ratio=16):
        super(CA, self).__init__()
        self.ave_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 3, 1, 1, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 3, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc2(self.relu1(self.fc1(self.ave_pool(x))))
        return self.sigmoid(out)

class DECODER(nn.Module):
    def __init__(self):
        super(DECODER, self).__init__()
        self.cbr7 = ConvBnRelu(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cbr8 = ConvBnRelu(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.erf3 = ERF()
        self.erf2 = ERF()
        self.erf1 = ERF()
        self.outconv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.outconv1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.outconv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.outconv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, Fcs, Fss, C3, C2, C1):
        """
        Fcs : 22*22*1
        Fss : 22*22*1
        C3 : 22*22*128
        C2 : 44*44*128
        C1 : 88*88*128
        """
        cs_48 = F.interpolate(Fcs, size=C2.size()[2:], mode='bilinear', align_corners=True)
        cs_96 = F.interpolate(Fcs, size=C1.size()[2:], mode='bilinear', align_corners=True)
        ss_48 = F.interpolate(Fss, size=C2.size()[2:], mode='bilinear', align_corners=True)
        ss_96 = F.interpolate(Fss, size=C1.size()[2:], mode='bilinear', align_corners=True)

        p3_ = self.erf3(Fcs, Fss, C3)
        p2_ = self.erf2(cs_48, ss_48, C2)
        p1_ = self.erf1(cs_96, ss_96, C1)

        out1 = self.cbr7(torch.concat([self.up(p3_), p2_], 1))
        ps_ = self.cbr8(torch.concat([self.up(out1), p1_], 1))

        ps_ = F.interpolate(ps_, size=(384, 384), mode='bilinear', align_corners=True)
        p3_ = F.interpolate(p3_, size=(384, 384), mode='bilinear', align_corners=True)
        p2_ = F.interpolate(p2_, size=(384, 384), mode='bilinear', align_corners=True)
        p1_ = F.interpolate(p1_, size=(384, 384), mode='bilinear', align_corners=True)
        ps = self.outconv(ps_)
        p3 = self.outconv1(p3_)
        p2 = self.outconv2(p2_)
        p1 = self.outconv3(p1_)
        return [ps, p3, p2, p1]

class DSINet(nn.Module):
    def __init__(self):
        super(DSINet, self).__init__()
        # capsnet
        self.capsnet = PWV(128)
        #self.hr = hrnet
        self.swin = swin
        self.proj1 = nn.Linear(128, 128)
        self.proj2 = nn.Linear(256, 128)
        self.proj3 = nn.Linear(512, 128)
        self.proj4 = nn.Linear(1024, 128)

        self.mce4 = MCE(128, 128)
        self.mce3 = MCE(128, 128)
        self.mce2 = MCE(128, 128)
        self.mce1 = MCE(128, 128)

        self.afe = AFE()
        self.decoder = DECODER()
        self.sim = SIM(1, 1)
        self.dropout = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):

        # layer = self.hr(x)
        # f1 = layer[0]
        # f2 = layer[1]
        # f3 = layer[2]
        # f4 = layer[3]
        f1, f2, f3, f4 = self.swin(x)
        _, _, H, W = x.shape
        B, _, C = f1.shape
        f1 = self.proj1(f1).view(B, H // 4, W // 4, 128).permute(0, 3, 1, 2)
        f2 = self.proj2(f2).view(B, H // 8, W // 8, 128).permute(0, 3, 1, 2)
        f3 = self.proj3(f3).view(B, H // 16, W // 16, 128).permute(0, 3, 1, 2)
        f4 = self.proj4(f4).view(B, H // 32, W // 32, 128).permute(0, 3, 1, 2)

        f4m = self.mce4(f4)
        f3m = self.mce3(f3)
        f2m = self.mce2(f2)
        f1m = self.mce1(f1)

        C1, C2, C3, cs = self.afe(f1m, f2m, f3m, f4m)

        # contrast semantic branch
        cs = self.dropout(cs)
        cs = torch.sigmoid(cs)

        # spatial structure semantic branch
        ss, pose = self.capsnet(C1)
        ss = self.dropout(ss)
        ss = torch.sigmoid(ss)
        # semantic interaction module
        cs_, ss_ = self.sim(cs, ss)
        Fcs = torch.sigmoid(cs_)
        Fss = torch.sigmoid(ss_)

        predict = self.decoder(Fcs, Fss, C3, C2, C1)
        return predict
