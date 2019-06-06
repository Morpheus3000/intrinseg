import torch
import torch.nn as nn
import torch.nn.modules.normalization as norm
import torch.nn.functional as F


class DirectIntrinsicsSN(nn.Module):

    #targets: array of {color, class}
    def __init__(self, input_channels, targets, deconv_param=False):
        super(DirectIntrinsicsSN, self).__init__()

        self.input_channels = input_channels
        self.targets = targets
        self.num_targets = len(targets)
        self.deconv_param=deconv_param

        # Encoder layers
        self.conv0 = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )


        # Decoder layers
        self.mid = nn.ModuleList()
        for i in range(self.num_targets):
            deconv = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
            self.mid.append(deconv)

        if 'class' in self.targets and self.deconv_param:
            deconv_index = self.targets.index("class")
        else:
            deconv_index = -1

        self.deconv0 = nn.ModuleList()
        for i in range(self.num_targets):
            learn_deconv = i == deconv_index
            deconv = nn.Sequential(
                nn.Conv2d(256*(self.num_targets+1), 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                self.upsample_layer(learn_deconv, num_channels=256)
            )
            self.deconv0.append(deconv)

        self.deconv1 = nn.ModuleList()
        for i in range(self.num_targets):
            learn_deconv = i == deconv_index
            deconv = nn.Sequential(
                nn.Conv2d(256*(self.num_targets+1), 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                self.upsample_layer(learn_deconv, num_channels=128)
            )
            self.deconv1.append(deconv)

        self.deconv2 = nn.ModuleList()
        for i in range(self.num_targets):
            learn_deconv = i == deconv_index
            deconv = nn.Sequential(
                nn.Conv2d(128*(self.num_targets+1), 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                self.upsample_layer(learn_deconv, num_channels=64)
            )
            self.deconv2.append(deconv)

        self.deconv3 = nn.ModuleList()
        for i in range(self.num_targets):
            learn_deconv = i == deconv_index
            deconv = nn.Sequential(
                nn.Conv2d(64*(self.num_targets+1), 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                self.upsample_layer(learn_deconv, num_channels=32)
            )
            self.deconv3.append(deconv)

        self.deconv4 = nn.ModuleList()
        for i in range(self.num_targets):
            learn_deconv = i == deconv_index
            deconv = nn.Sequential(
                nn.Conv2d(32*(self.num_targets+1), 16, 3, 1, 1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                self.upsample_layer(learn_deconv, num_channels=16)
            )
            self.deconv4.append(deconv)

        output_dims = [self.output_dim(target) for target in self.targets]

        self.output = nn.ModuleList()
        for i in range(self.num_targets):
            deconv = nn.Sequential(
                nn.Conv2d(16*(self.num_targets+1), 16, 3, 1, 1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Conv2d(16, output_dims[i], 3, 1, 1),
                nn.BatchNorm2d(output_dims[i]),
                nn.ReLU(True)
            )
            self.output.append(deconv)


    def output_dim(self, target):
        if target == 'color':
            return 3
        elif target == 'single':
            return 1
        elif target == 'class':
            return 16
        elif target == 'class-tb':
            return 12

    def upsample_layer(self, learn_deconv, num_channels=0):
        if learn_deconv:
            return nn.ConvTranspose2d(num_channels, num_channels, 3, stride=2, padding=1, output_padding=1)
        else:
            return nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        ##m2 = self.mid[1](x5)
        #xmcat = [xm1, xm2, x5]
        xmid = torch.cat([deconv(x5) for deconv in self.mid] + [x5], 1)

        x0d = torch.cat([deconv(xmid) for deconv in self.deconv0] + [x4], 1)
        x1d = torch.cat([deconv(x0d) for deconv in self.deconv1] + [x3], 1)
        x2d = torch.cat([deconv(x1d) for deconv in self.deconv2] + [x2], 1)
        x3d = torch.cat([deconv(x2d) for deconv in self.deconv3] + [x1], 1)
        x4d = torch.cat([deconv(x3d) for deconv in self.deconv4] + [x0], 1)

        x_out = [deconv(x4d) for deconv in self.output]

        if not self.training and 'class' in self.targets:
            seg_index = self.targets.index("class")
            x_out[seg_index] = F.softmax(x_out[seg_index], dim=1)

        # print("x0 shape: %s" % str(list(x0.shape)))
        # print("x1 shape: %s" % str(list(x1.shape)))
        # print("x2 shape: %s" % str(list(x2.shape)))
        # print("x3 shape: %s" % str(list(x3.shape)))
        # print("x4 shape: %s" % str(list(x4.shape)))
        # print("x5 shape: %s" % str(list(x5.shape)))
        # print("xmd shape: %s" % str(list(xmid.shape)))
        # print("x0d shape: %s" % str(list(x0d.shape)))
        # print("x1d shape: %s" % str(list(x1d.shape)))
        # print("x2d shape: %s" % str(list(x2d.shape)))
        # print("x3d shape: %s" % str(list(x3d.shape)))
        # print("x4d shape: %s" % str(list(x4d.shape)))
        # print("xou shape: %s" % str(list(x_out[0].shape)))

        if len(x_out) > 1:
            return tuple(x_out)
        else:
            return x_out[0]
