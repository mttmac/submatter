import torch
import torch.nn as nn
from pathlib import Path


def normalize_to_fp16(x):
    # Normalize channels to ImageNet distribution per paper and convert to fp16
    # Must be written to avoid use of ScatterND layers in ONNX, inspect at netron.app
    # MyriadX blobs do not support Scatter layers

    x = x.float()  # cast

    # Standardize to unit gaussian distribution
    mu = torch.mean(x, dim=(2, 3), keepdim=True)
    std = torch.std(x, dim=(2, 3), keepdim=True)
    x = (x - mu) / std

    # Scale to ImageNet input distribution
    img_mu = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    r, g, b = torch.split(x, 1, dim=1)
    r = (r * img_std[0]) + img_mu[0]
    g = (g * img_std[1]) + img_mu[1]
    b = (b * img_std[2]) + img_mu[2]
    x = torch.cat((r, g, b), dim=1)

    # Naive approach that results in a mess of a network that needs ScatterND
    # for c in range(x.shape[1]):
    #     xc = x[:, c]
    #     xc = torch.subtract(xc, xc.mean())
    #     xc = torch.div(xc, x[:, c].std())
    #     xc = torch.multiply(xc, ideal_std[c])
    #     x[:, c] = torch.add(xc, ideal_mean[c])

    return x


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)  # reshape
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)  # flatten channels

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=66):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc_y = nn.Linear(output_channels, num_classes)
        self.fc_p = nn.Linear(output_channels, num_classes)
        self.fc_r = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = normalize_to_fp16(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        y = self.fc_y(x)
        p = self.fc_p(x)
        r = self.fc_r(x)
        return y, p, r


def build_hopenet_lite():
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Loads weights from open source deep-head-pose-lite implementation
    """
    model = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], 66)
    net = torch.load(Path('ref/deep-head-pose-lite/model/shuff_epoch_120.pkl'), map_location=torch.device('cpu'))
    model.load_state_dict(net)

    return model

