import torch
from torch import nn


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Conv2d(dim, default(dim_out, dim), 3, 2, 1)
    )


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class MFAB(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1, bias=0):
        super(MFAB, self).__init__()

        hidden_features = 16

        self.project_in = nn.Conv2d(dim, hidden_features * 3, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 3, hidden_features * 3, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features * 3, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 3, hidden_features * 3, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features * 3, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden_features * 3, hidden_features *3, kernel_size=7, stride=1, padding=3,
                                   groups=hidden_features * 3, bias=bias)

        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu7 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 3, hidden_features, kernel_size=3, stride=1, padding=1,
                                     groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 3, hidden_features, kernel_size=5, stride=1, padding=2,
                                     groups=hidden_features , bias=bias)
        self.dwconv7x7_1 = nn.Conv2d(hidden_features * 3, hidden_features, kernel_size=7, stride=1, padding=3,
                                     groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()
        self.relu7_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 9, dim, kernel_size=1, bias=bias)

        self.attn = CBAMLayer(hidden_features * 9)

    def forward(self, x):
        shortcut = x.clone()
        x = self.project_in(x)
        x1_3, x2_3, x3_3 = self.relu3(self.dwconv3x3(x)).chunk(3, dim=1)
        x1_5, x2_5, x3_5 = self.relu5(self.dwconv5x5(x)).chunk(3, dim=1)
        x1_7, x2_7, x3_7 = self.relu5(self.dwconv7x7(x)).chunk(3, dim=1)

        x1 = torch.cat([x1_3, x1_5, x1_7], dim=1)
        x2 = torch.cat([x2_3, x2_5, x2_7], dim=1)
        x3 = torch.cat([x3_3, x3_5, x3_7], dim=1)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.attn(x)
        x = self.project_out(x)

        return x + shortcut

class MFAM(torch.nn.Module):
    def __init__(self, dim_in=3, n_feats=64):
        super(MFAM, self).__init__()

        self.first_conv = nn.Conv2d(dim_in, n_feats, 3,1,1)

        self.down = nn.ModuleList([])
        in_out_up = [(64, 128), (128, 256), (256, 256)]

        for ind, (dim_in, dim_out) in enumerate(in_out_up):
            is_last = ind == (len(in_out_up) - 1)
            self.down.append(nn.ModuleList([
                MFAB(dim_in),
                MFAB(dim_in),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        MFAM_list = []

        for MFAB1, MFAB2, downsample in self.down:
            x = MFAB1(x)
            x = MFAB2(x)
            MFAM_list.append(x)
            x = downsample(x)

        return MFAM_list
