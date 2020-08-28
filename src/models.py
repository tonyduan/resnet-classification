import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.datasets import *
from src.blocks import BasicBlock, Bottleneck, BasicBlockV2, BottleneckV2
from src.blocks import InvertedBottleneck


class Classifier(nn.Module):
    """
    Generic classifier module that sets up appropriate normalization and log-lik loss.
    """
    def __init__(self, dataset, device, precision="float"):
        super().__init__()
        self.norm = NormalizeLayer(get_normalization_shape(dataset),
                                   **get_normalization_stats(dataset))
        self.device = device
        self.precision = precision
        self.focal = False

    def initialize_weights(self):
        raise NotImplementedError

    def set_device_and_precision(self):
        for m in self.modules():
            m = m.to(self.device)
            if self.precision == "half":
                m = m.half()
            if self.precision == "float":
                m = m.float()
            if self.precision == "double":
                m = m.double()

    def forward(self, x):
        raise NotImplementedError

    def forecast(self, theta):
        return Categorical(logits=theta)

    def loss(self, x, y, sample_weights=None):
        forecast = self.forecast(self.forward(x))
        if self.focal:
            p = torch.gather(forecast.probs, 1, y.unsqueeze(1)).squeeze()
            nll = -forecast.log_prob(y) * (1 - p) ** 3.0
        else:
            nll = -forecast.log_prob(y)
        return nll * sample_weights if sample_weights is not None else nll

    def brier_loss(self, x, y, sample_weights=None):
        forecast_probs = self.forecast(self.forward(x)).probs
        one_hot_labels = torch.eye(forecast_probs.shape[1], device=x.device)[y]
        brier = torch.mean(torch.pow(forecast_probs - one_hot_labels, 2), dim=1)
        return brier * sample_weights if sample_weights is not None else brier


class DataParallelWrapper(nn.DataParallel):
    """
    Wrapper around nn.DataParallel that exposes custom methods.
    Source: https://github.com/pytorch/pytorch/issues/16885
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class NormalizeLayer(nn.Module):
    """
    Normalizes across the first non-batch axis.

    Examples
    --------
        (64, 3, 32, 32) [CIFAR] => normalizes across channels.
        (64, 8)         [UCI]   => normalizes across features
    """
    def __init__(self, dim, mu, sigma):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu).reshape(dim), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(sigma).reshape(dim), requires_grad=False)

    def forward(self, x):
        return (x - self.mu) / self.sigma


class LinearModel(Classifier):
    """
    Straightforward linear model where logits are a linear combination of features (convex).
    """
    def __init__(self, dataset, device, precision):
        super().__init__(dataset, device, precision)
        self.fc = nn.Linear(get_dim(dataset), get_num_labels(dataset), bias=True)
        self.flatten = nn.Flatten()
        self.set_device_and_precision()

    def forward(self, x):
        return self.fc(self.flatten(self.norm(x)))

    def class_activation_map(self, x, y):
        cam = self.fc.weight[y] * self.flatten(x)
        return F.interpolate(cam.reshape((x.shape[0], 3, 32, 32)), scale_factor=0.25).mean(dim=1)


class ResNet(Classifier):
    """
    Classic residual network [He et al. CVPR 2016].
    See `blocks.py` for BasicBlock and Bottleneck implementations.

    Also supports re-ordered "pre-activation" residual network [He et al. ECCV 2016].
    i.e. (conv => batch norm => activation) replaced with (batch norm => activation => conv).
    In this case we need to use BasicBlockV2 or BottleneckV2.

    Consists of:
        1. Initial convolution, batch norm, then max pool (strided if ImageNet).
        2. Sequence of layers, each consisting of a number of blocks with fixed number of filters.
           At the first convolution of each layer we downsample by half by setting stride = 2.
        3. Global average pool, then linear layer to logits.

    Configurations for canonical models are described below:
    - ResNets, for both CIFAR-10 and ImageNet [He et al. CVPR 2016]
    - Wide ResNets [Zagoruyko and Komodakis ECCV 2016].
    - SE ResNets [Hu et al. CVPR 2018].
    """
    # == ImageNet models
    resnet18_layers = [
        {"block": BasicBlock, "num_blocks": 2, "num_filters": 64, "squeeze": 4},  # 64 x 56 x 56 output
        {"block": BasicBlock, "num_blocks": 2, "num_filters": 128, "squeeze": 4}, # 128 x 28 x 28 output
        {"block": BasicBlock, "num_blocks": 2, "num_filters": 256, "squeeze": 4}, # 256 x 14 x 14 output
        {"block": BasicBlock, "num_blocks": 2, "num_filters": 512, "squeeze": 4}, # 512 x 7 x 7 output
    ]
    resnet50_layers = [
        {"block": Bottleneck, "num_blocks": 3, "num_filters": 256, "squeeze": 4},  # 256 x 56 x 56 output
        {"block": Bottleneck, "num_blocks": 4, "num_filters": 512, "squeeze": 4},  # 512 x 28 x 28 output
        {"block": Bottleneck, "num_blocks": 6, "num_filters": 1024, "squeeze": 4}, # 1024 x 14 x 14 output
        {"block": Bottleneck, "num_blocks": 3, "num_filters": 2048, "squeeze": 4}, # 2048 x 7 x 7 output
    ]
    resnext50_32x4d_layers = [
        {"block": Bottleneck, "num_blocks": 3, "num_filters": 256, "groups": 32, "squeeze": 2},
        {"block": Bottleneck, "num_blocks": 4, "num_filters": 512, "groups": 32, "squeeze": 2},
        {"block": Bottleneck, "num_blocks": 6, "num_filters": 1024, "groups": 32, "squeeze": 2},
        {"block": Bottleneck, "num_blocks": 3, "num_filters": 2048, "groups": 32, "squeeze": 2},
    ]
    se_resnet50_layers = [
        {"block": Bottleneck, "num_blocks": 3, "num_filters": 256, "squeeze": 4, "se": True},  # 256 x 56 x 56 output
        {"block": Bottleneck, "num_blocks": 4, "num_filters": 512, "squeeze": 4, "se": True},  # 512 x 28 x 28 output
        {"block": Bottleneck, "num_blocks": 6, "num_filters": 1024, "squeeze": 4, "se": True}, # 1024 x 14 x 14 output
        {"block": Bottleneck, "num_blocks": 3, "num_filters": 2048, "squeeze": 4, "se": True}, # 2048 x 7 x 7 output
    ]

    # == CIFAR-10 models
    resnet110_layers = [
        {"block": BasicBlock, "num_blocks": 18, "num_filters": 16}, # 16 x 32 x 32 output
        {"block": BasicBlock, "num_blocks": 18, "num_filters": 32}, # 32 x 16 x 16 output
        {"block": BasicBlock, "num_blocks": 18, "num_filters": 64}, # 64 x 8 x 8 output
    ]
    se_resnet110_layers = [
        {"block": BasicBlock, "num_blocks": 18, "num_filters": 16, "se": True}, # 16 x 32 x 32 output
        {"block": BasicBlock, "num_blocks": 18, "num_filters": 32, "se": True}, # 32 x 16 x 16 output
        {"block": BasicBlock, "num_blocks": 18, "num_filters": 64, "se": True}, # 64 x 8 x 8 output
    ]
    wrn_40_2_layers = [
        {"block": BasicBlockV2, "num_blocks": 6, "num_filters": 32},  # 32 x 32 x 32 output
        {"block": BasicBlockV2, "num_blocks": 6, "num_filters": 64},  # 64 x 16 x 16 output
        {"block": BasicBlockV2, "num_blocks": 6, "num_filters": 128}, # 128 x 8 x 8 output
    ]
    wrn_28_10_layers = [
        {"block": BasicBlockV2, "num_blocks": 4, "num_filters": 160}, # 160 x 32 x 32 output
        {"block": BasicBlockV2, "num_blocks": 4, "num_filters": 320}, # 320 x 16 x 16 output
        {"block": BasicBlockV2, "num_blocks": 4, "num_filters": 640}, # 640 x 8 x 8 output
    ]

    def __init__(self, dataset, device, precision, layers_config=wrn_40_2_layers,
                 norm_layer="batch_norm", focal=False):

        super().__init__(dataset, device, precision)
        self.pre_activation = layers_config[0]["block"].pre_activation
        self.focal = focal

        if norm_layer == "batch_norm":
            norm_layer = nn.BatchNorm2d
        elif norm_layer == "group_norm":
            norm_layer=lambda out_filters: nn.GroupNorm(out_filters // 16, out_filters)
        else:
            raise ValueError

        if dataset == "imagenet":
            num_filters = 64
            self.conv1 = nn.Conv2d(3, num_filters, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            num_filters = 16
            self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()

        self.bn_init = norm_layer(num_filters)
        self.blocks = nn.ModuleList()

        for layer_no, config in enumerate(layers_config):
            for block_no in range(config["num_blocks"]):
                stride = 2 if layer_no != 0 and block_no == 0 else 1
                self.blocks.append(config["block"](in_filters=num_filters,
                                                   out_filters=config["num_filters"],
                                                   stride=stride,
                                                   norm_layer=norm_layer,
                                                   **config))
                num_filters = config["num_filters"]

        self.bn_final = norm_layer(num_filters) if self.pre_activation else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_filters, get_num_labels(dataset))
        self.initialize_weights()
        self.set_device_and_precision()

    def initialize_weights(self):
        """
        Initialize as in [He et al. ICCV 2015]:
        1. Convolution weights ~ N(0, 2 / (kernel_size x kernel_size x n_output_filters))
        2. Batch norm weights = 1 biases = 0.
        Note that this guarantees the output variance is O(1) for deep networks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.norm(x)
        out = self.conv1(out)
        out = self.bn_init(out)
        out = F.relu(out)
        out = self.maxpool(out)
        for block in self.blocks:
            out = block(out)
        out = self.bn_final(out)
        out = F.relu(out)
        out = self.avgpool(out)
        out = self.linear(self.flatten(out))
        return out

    def class_activation_map(self, x, y):
        out = self.norm(x)
        out = self.conv1(out)
        out = self.bn_init(out)
        out = F.relu(out)
        out = self.maxpool(out)
        for block in self.blocks:
            out = block(out)
        out = self.bn_final(out)
        out = F.relu(out)
        cam = self.linear.weight[y].unsqueeze(1) @ out.flatten(start_dim=2, end_dim=-1)
        cam = cam.squeeze(1) + self.linear.bias[y].unsqueeze(1)
        cam = cam.view(-1, *out.shape[2:])
        return cam


class WRN_40_2(ResNet):
    def __init__(self, dataset, device, precision):
        super().__init__(dataset, device, precision, layers_config=ResNet.wrn_40_2_layers)


class WRN_28_10(ResNet):
    def __init__(self, dataset, device, precision):
        super().__init__(dataset, device, precision, layers_config=ResNet.wrn_28_10_layers)


class MobileNetV2(Classifier):
    """
    Classic residual network [Sandler et al. CVPR 2018].
    See `blocks.py` for InvertedBottleneck implementation.

    Consists of:
        1. Initial convolution, batch norm (strided for ImageNet).
        2. Sequence of layers, each consisting of a number of blocks with fixed number of filters.
           At the first convolution of each layer we may downsample by half by setting stride = 2.
        3. Global average pool, then linear layer to logits.

    Configurations for canonical models are described below:
    - MobileNetV2 [Sandler et al. CVPR 2018].
    """
    # == ImageNet models
    mobilenetv2_layers = [
        {"block": InvertedBottleneck, "num_blocks": 1, "num_filters": 16, "firststride": 1, "expand": 1}, # 16 x 112 x 112 output
        {"block": InvertedBottleneck, "num_blocks": 2, "num_filters": 24, "firststride": 2, "expand": 6}, # 24 x 56 x 56 output
        {"block": InvertedBottleneck, "num_blocks": 3, "num_filters": 32, "firststride": 2, "expand": 6}, # 32 x 28 x 28 output
        {"block": InvertedBottleneck, "num_blocks": 4, "num_filters": 64, "firststride": 2, "expand": 6}, # 64 x 14 x 14 output
        {"block": InvertedBottleneck, "num_blocks": 3, "num_filters": 96, "firststride": 1, "expand": 6}, # 96 x 14 x 14 output
        {"block": InvertedBottleneck, "num_blocks": 3, "num_filters": 160, "firststride": 2, "expand": 6}, # 160 x 7 x 7 output
        {"block": InvertedBottleneck, "num_blocks": 1, "num_filters": 320, "firststride": 1, "expand": 6}, # 320 x 7 x 7 output
    ]

    def __init__(self, dataset, device, precision, layers_config=mobilenetv2_layers):

        super().__init__(dataset, device, precision)

        num_filters = 32
        self.conv_init = nn.Conv2d(3, num_filters, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(num_filters)
        self.blocks = nn.ModuleList()

        for layer_no, config in enumerate(layers_config):
            for block_no in range(config["num_blocks"]):
                stride = config["firststride"] if layer_no != 0 and block_no == 0 else 1
                self.blocks.append(config["block"](in_filters=num_filters,
                                                   out_filters=config["num_filters"],
                                                   stride=stride,
                                                   **config))
                num_filters = config["num_filters"]

        self.conv_final = nn.Conv2d(num_filters, num_filters * 4, kernel_size=1, bias=False)
        self.bn_final = nn.BatchNorm2d(num_filters * 4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_filters, get_num_labels(dataset))
        self.initialize_weights()
        self.set_device_and_precision()

    def initialize_weights(self):
        """
        Initialize as in [He et al. ICCV 2015]:
        1. Convolution weights ~ N(0, 2 / (kernel_size x kernel_size x n_output_filters))
        2. Batch norm weights = 1 biases = 0.
        Note that this guarantees the output variance is O(1) for deep networks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.norm(x)
        out = self.conv_init(out)
        out = self.bn_init(out)
        out = F.relu6(out)
        for block in self.blocks:
            out = block(out)
        out = self.conv_final(out)
        out = self.bn_final(out)
        out = F.relu6(out)
        out = self.avgpool(out)
        out = self.linear(self.flatten(out))
        return out

