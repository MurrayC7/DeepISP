# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.autograd import Variable
from .unet_parts import *
import math
import time
import numpy as np

def weights_init(modules, type='xavier'):
    m = modules
    if isinstance(m, nn.Conv2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            m.weight.data.fill_(1.0)

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Module):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    m.weight.data.fill_(1.0)

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False)

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pf,pad, dilation):
        super(BasicBlock, self).__init__()
        width = int(math.floor(planes * (26.0 / 64.0)))
        self.conv1 = nn.Sequential(convbn(inplanes, width * 4, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        convs = []
        for i in range(3):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=pad,dilation=dilation, bias=False))
        self.convs = nn.ModuleList(convs)

        self.conv2 = convbn(width * 4, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.pf = pf
        self.stride = stride
        self.width = width
    def forward(self, x):
        out = self.conv1(x)

        spx = torch.split(out, self.width, 1)
        for i in range(3):
            if i == 0 or self.downsample is not None:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = F.relu(sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        temp = self.pf(spx[3])
        out = torch.cat((out, temp), 1)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = F.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(1, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 7, 1, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 2, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 128, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn(128, 32, 1, 1, 0, 1)
                                    )

        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
                                nn.Linear(32, 8),
                                nn.ReLU(inplace=True),
                                nn.Linear(8, 32),
                                nn.Sigmoid()
                               )
        # self.weight1 = nn.Sequential(convbn(32, 32, 3, 1, 16, 16),
        #                              nn.ReLU(inplace=True),
        #                              convbn(32, 32, 3, 1, 1, 1),
        #                              nn.Sigmoid())

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 128, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn(128, 32, 1, 1, 0, 1)
                                     )
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc2 = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 32),
            nn.Sigmoid()
        )
        # self.weight2 = nn.Sequential(convbn(32, 32, 3, 1, 8, 8),
        #                              nn.ReLU(inplace=True),
        #                              convbn(32, 32, 3, 1, 1, 1),
        #                              nn.Sigmoid())

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 128, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn(128, 32, 1, 1, 0, 1)
                                     )
        self.avg_pool3 = nn.AdaptiveAvgPool2d(1)
        self.fc3 = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 32),
            nn.Sigmoid()
        )
        # self.weight3 = nn.Sequential(convbn(32, 32, 3, 1, 4, 4),
        #                              nn.ReLU(inplace=True),
        #                              convbn(32, 32, 3, 1, 1, 1),
        #                              nn.Sigmoid())

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 128, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn(128, 32, 1, 1, 0, 1)
                                     )

        self.avg_pool4 = nn.AdaptiveAvgPool2d(1)
        self.fc4 = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 32),
            nn.Sigmoid()
        )
        # self.weight4 = nn.Sequential(convbn(32, 32, 3, 1, 2, 2),
        #                              nn.ReLU(inplace=True),
        #                              convbn(32, 32, 3, 1, 1, 1),
        #                              nn.Sigmoid())

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False),
                                      nn.ReLU(inplace=True))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        pf = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False)
           pf = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pf,pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pf,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output  = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output_skip = self.layer3(output_raw)

        output_branch1 = self.branch1(output_skip)

        b1,c1,_,_ = output_branch1.size()
        y1 = self.avg_pool1(output_branch1).view(b1, c1)
        y1 = self.fc1(y1).view(b1, c1, 1, 1)
        output_branch1 = output_branch1*y1 +output_branch1
        # output_branch1 = self.weight1(output_branch1)*output_branch1

        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                    align_corners=True)

        output_branch2 = self.branch2(output_skip)

        b2, c2, _, _ = output_branch2.size()
        y2 = self.avg_pool2(output_branch2).view(b2, c2)
        y2 = self.fc2(y2).view(b2, c2, 1, 1)
        output_branch2 = output_branch2 * y2+output_branch2
        # output_branch2 = self.weight2(output_branch2) * output_branch2

        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                    align_corners=True)

        output_branch3 = self.branch3(output_skip)

        b3, c3, _, _ = output_branch3.size()
        y3 = self.avg_pool3(output_branch3).view(b3, c3)
        y3 = self.fc3(y3).view(b3, c3, 1, 1)
        output_branch3 = output_branch3 * y3+output_branch3
        # output_branch3 = self.weight3(output_branch3) * output_branch3

        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                    align_corners=True)

        output_branch4 = self.branch4(output_skip)

        b4, c4, _, _ = output_branch4.size()
        y4 = self.avg_pool4(output_branch4).view(b4, c4)
        y4 = self.fc4(y4).view(b4, c4, 1, 1)
        output_branch4 = output_branch4 * y4+output_branch4
        # output_branch4 = self.weight4(output_branch4) * output_branch4

        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                    align_corners=True)

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)

        output_feature = self.lastconv(output_feature)

        return output_feature

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=(1,2,2), pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=(1,2,2), pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=(0,1,1), stride=(1,2,2),
                               bias=False))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=(0,1,1), stride=(1,2,2),
                               bias=False))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.out = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(96, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classify1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=1, padding=0, stride=1, bias=False),
                                      # nn.ReLU(inplace=True)
                                      nn.Tanh()
                                      )
        self.classify2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=1, padding=0, stride=1, bias=False),
                                      # nn.ReLU(inplace=True)
                                      nn.Tanh()
                                      )
        self.classify3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=1, padding=0, stride=1, bias=False),
                                      # nn.ReLU(inplace=True)
                                      nn.Tanh()
                                      )


        weights_init(self.modules(),'xavier')
    def forward(self, inr,ing,inb):

        feasR = self.out(inr)
        feasG = self.out(ing)
        feasB = self.out(inb)

        cost = Variable(torch.FloatTensor(feasR.size()[0], feasR.size()[1]*3, 3, feasR.size()[2],feasR.size()[3]).zero_()).cuda()
        cost[:,:feasR.size()[1],0,:,:] = feasR
        cost[:, feasR.size()[1]:feasR.size()[1]*2, 0, :, :] = feasG
        cost[:, feasR.size()[1]*2:feasR.size()[1]*3, 0, :, :] = feasB

        cost[:, :feasR.size()[1], 1, :, :] = feasB
        cost[:, feasR.size()[1]:feasR.size()[1] * 2, 1, :, :] = feasR
        cost[:, feasR.size()[1] * 2:feasR.size()[1] * 3, 1, :, :] = feasG

        cost[:, :feasR.size()[1], 2, :, :] = feasG
        cost[:, feasR.size()[1]:feasR.size()[1] * 2, 2, :, :] = feasB
        cost[:, feasR.size()[1] * 2:feasR.size()[1] * 3, 2, :, :] = feasR

        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0)+cost0


        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classify1(out1)
        cost2 = self.classify2(out2) + cost1
        cost3 = self.classify3(out3) + cost2

        if self.training:
            pred1 = torch.squeeze(cost1, 1)
            pred2 = torch.squeeze(cost2, 1)

        pred3 = torch.squeeze(cost3, 1)

        if self.training:
            return pred3, pred2, pred1
        else:
            return pred3


