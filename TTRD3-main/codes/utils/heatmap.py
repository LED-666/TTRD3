# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import os
#
# def heatmap(feature_maps, path):
#     # 假设 feature_maps 是你的Tensor特征图，它的shape可能是[B, C, H, W]，其中B是批次大小，C是通道数，H是高度，W是宽度
#     # 这里我们创建一个随机的特征图作为示例
#     feature_maps = feature_maps  # 示例中B=1，C=10，H=28，W=28
#
#     feature_maps_cpu = feature_maps.cpu()
#
#     # 将Tensor转换为numpy数组
#     feature_maps_np = feature_maps_cpu.squeeze(0).detach().numpy()
#
#     # 将所有通道的特征图相加
#     summed_feature_maps = np.sum(feature_maps_np, axis=0)
#
#     # 归一化叠加后的特征图
#     summed_feature_maps = (summed_feature_maps - summed_feature_maps.min()) / (
#                 summed_feature_maps.max() - summed_feature_maps.min())
#
#     # 显示热力图
#     plt.imshow(summed_feature_maps, cmap='viridis')
#     plt.axis('off')
#     # plt.colorbar()  # 显示颜色条
#
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#     # 保存热力图到文件
#     plt.savefig(f'{path}/heatmap.png', bbox_inches='tight', pad_inches=0)
#     print("heatmap")
#
#
#
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.exposure import equalize_hist

def heatmap(feature_maps, path):
    feature_maps = feature_maps.cpu().squeeze(0).detach().numpy()
    summed_feature_maps = np.sum(feature_maps, axis=0)

    # 使用分位数裁剪极值，增强对比度
    min_val = np.percentile(summed_feature_maps, 0.5)  # 5% 分位数
    max_val = np.percentile(summed_feature_maps, 99.5)  # 95% 分位数

    # 归一化到 [0, 1]，但使用分位数作为范围
    summed_feature_maps = (summed_feature_maps - min_val) / (max_val - min_val)
    summed_feature_maps = np.clip(summed_feature_maps, 0, 1)  # 确保值在 [0, 1] 范围内

    # 显示热力图
    plt.imshow(summed_feature_maps, cmap='viridis')  # 使用默认的 'viridis' 颜色映射
    plt.axis('off')

    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f'{path}/heatmap.png', bbox_inches='tight', pad_inches=0)
    print("heatmap saved")