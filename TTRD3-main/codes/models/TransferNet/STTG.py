import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEncoding(nn.Module):
    """ 位置编码模块：增强位置感知 """

    def __init__(self, in_channels, max_size=256):
        super().__init__()
        x = torch.linspace(-1, 1, max_size)
        y = torch.linspace(-1, 1, max_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('grid', torch.stack([grid_x, grid_y], dim=0))  # [2, H, W]
        self.pos_conv = nn.Conv2d(2, in_channels, kernel_size=3, padding=1)
        nn.init.normal_(self.pos_conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.pos_conv.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()
        pos_grid = F.interpolate(self.grid.unsqueeze(0), size=(h, w), mode='bilinear')  # [1,2,H,W]
        pos_feat = self.pos_conv(pos_grid.expand(b, -1, -1, -1))  # [B,C,H,W]
        return x + pos_feat  # 残差连接


class SEBlock(nn.Module):
    """ 通道注意力模块 """

    def __init__(self, channel, reduction=16):
        super().__init__()
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


class DynamicKSelector(nn.Module):
    """ 动态K值选择器 """

    def __init__(self, max_k=5, num_levels=3):
        super().__init__()
        self.max_k = max_k
        self.num_levels = num_levels
        self.k_logits = nn.Parameter(torch.randn(num_levels, max_k))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, R_combined):
        N, HrWr, HW = R_combined.shape
        k_probs = F.gumbel_softmax(self.k_logits, tau=self.temperature, dim=-1)
        k_hard = torch.argmax(k_probs, dim=-1) + 1  # K ∈ [1, max_k]
        k_soft = torch.sum(k_probs * torch.arange(1, self.max_k + 1, device=k_probs.device), dim=-1)
        k_selected = k_hard.detach() + k_soft - k_soft.detach()
        topk_weights = []
        for lvl in range(self.num_levels):
            k = int(k_hard[lvl].item())
            R_star, R_idx = torch.topk(R_combined, k, dim=1)
            pad_size = self.max_k - k
            R_star_padded = F.pad(R_star, (0, 0, 0, pad_size), value=-float('inf'))
            R_idx_padded = F.pad(R_idx, (0, 0, 0, pad_size), value=0)
            topk_weights.append((R_star_padded, R_idx_padded))
        return k_selected, topk_weights


class STTG(nn.Module):
    """ 搜索与迁移模块 """

    def __init__(self, max_k=5, reduction=16):
        super().__init__()
        self.max_k = max_k
        self.k_selector = DynamicKSelector(max_k=max_k, num_levels=3)
        self.se_lv1 = SEBlock(64, reduction)
        self.se_lv2 = SEBlock(128, reduction)
        self.se_lv3 = SEBlock(256, reduction)
        self.global_weight = nn.Parameter(torch.tensor(0.1))

        # 位置编码器：仅对参考图像特征编码
        self.ref_pos_lv1 = PositionEncoding(64, max_size=256)
        self.ref_pos_lv2 = PositionEncoding(128, max_size=128)
        self.ref_pos_lv3 = PositionEncoding(256, max_size=64)

    def bis(self, input, dim, index):
        """ Batch-wise索引选择 """
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
        # === 步骤1：结构相似性计算（lr与refsr_lv3匹配） ===
        N, C, H, W = lrsr_lv3.shape
        HW = H * W

        # 对refsr_lv3添加位置编码
        refsr_lv3_enc = self.ref_pos_lv3(refsr_lv3)  # [N,256,64,64]

        # 局部相关性
        lrsr_unfold = F.unfold(lrsr_lv3, kernel_size=3, padding=1)  # [N, 2304, 4096]
        refsr_unfold = F.unfold(refsr_lv3_enc, kernel_size=3, padding=1)  # [N, 2304, 4096]
        refsr_norm = F.normalize(refsr_unfold.permute(0, 2, 1), dim=2)  # [N, 4096, 2304]
        lrsr_norm = F.normalize(lrsr_unfold, dim=1)  # [N, 2304, 4096]
        R_local = torch.bmm(refsr_norm, lrsr_norm)  # [N, 4096, 4096]

        # 全局相关性
        ref_global = F.normalize(refsr_lv3_enc.mean(dim=[2, 3]), dim=1)  # [N,256]
        lr_global = F.normalize(lrsr_lv3.mean(dim=[2, 3]), dim=1)  # [N,256]
        R_global = torch.bmm(ref_global.unsqueeze(1), lr_global.unsqueeze(2)).expand_as(R_local)

        R_combined = (1 - self.global_weight) * R_local + self.global_weight * R_global

        # === 步骤2：动态选择Top-K匹配区域 ===
        k_selected, topk_weights = self.k_selector(R_combined)

        # === 步骤3：从原始高清参考图像ref迁移细节 ===
        def transfer_level(ref_feat, level_idx, kernel, stride, pad):
            """ 从原始高清参考图像迁移特征 """
            R_star, R_idx = topk_weights[level_idx]
            k = int(k_selected[level_idx].item())

            # 对原始高清参考特征添加位置编码
            if level_idx == 0:
                ref_feat = self.ref_pos_lv3(ref_feat)
            elif level_idx == 1:
                ref_feat = self.ref_pos_lv2(ref_feat)
            else:
                ref_feat = self.ref_pos_lv1(ref_feat)

            # 展开特征
            ref_unfold = F.unfold(ref_feat, kernel_size=kernel, stride=stride, padding=pad)
            T_all = torch.zeros(N, ref_unfold.size(1), HW, device=ref_feat.device)

            for i in range(k):
                idx = R_idx[:, i]  # [N, HW]
                selected = self.bis(ref_unfold, 2, idx)  # [N, C*k*k, HW]
                weight = torch.sigmoid(R_star[:, i]).unsqueeze(1)
                T_all += selected * weight

            # 折叠回特征图（显式指定参数）
            output_size = (H * stride, W * stride)
            T = F.fold(
                T_all,
                output_size=output_size,
                kernel_size=kernel,
                stride=stride,
                padding=pad
            ) / (kernel ** 2)
            return T

        # 各层级迁移（使用原始高清参考图像ref）
        T_lv3 = transfer_level(ref_lv3, 0, 3, 1, 1)  # Level3: kernel=3, stride=1, pad=1
        T_lv2 = transfer_level(ref_lv2, 1, 6, 2, 2)  # Level2: kernel=6, stride=2, pad=2
        T_lv1 = transfer_level(ref_lv1, 2, 12, 4, 4)  # Level1: kernel=12, stride=4, pad=4

        # 通道注意力增强
        T_lv3 = self.se_lv3(T_lv3)
        T_lv2 = self.se_lv2(T_lv2)
        T_lv1 = self.se_lv1(T_lv1)

        return T_lv3, T_lv2, T_lv1