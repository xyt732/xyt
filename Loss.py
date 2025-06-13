# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy import ndimage
import jittor as jt
import jittor.nn as nn
import matplotlib.pyplot as plt
from PIL import Image


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, label_file="train_label.txt", w0=10, sigma=512*0.08, save_dir="./weight_maps", eps=1e-6):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.w0 = w0
        self.sigma = sigma
        self.sigma_sq = 2 * sigma * sigma
        self.save_dir = save_dir
        self.eps = eps  # 数值稳定系数

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 加载标签文件列表
        self.label_paths = []
        with open(label_file, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    self.label_paths.append(path)

        # 预计算并保存所有权重图
        self.precompute_all_weights()

        # 创建文件名到权重图的映射
        self.weights_cache = {}
        for full_path in self.label_paths:
            filename = os.path.basename(full_path)  #从完整文件路径中提取文件名
            weight_path = os.path.join(save_dir, f"{filename.replace('.png', '')}.npy")
            if os.path.exists(weight_path): #如果权重图存在
                self.weights_cache[filename] = weight_path

    def precompute_all_weights(self):
        """预计算并保存所有训练labels权重图"""
        for full_path in self.label_paths:
            filename = os.path.basename(full_path)  #从完整文件路径中提取文件名
            weight_path = os.path.join(self.save_dir, f"{filename.replace('.png', '')}.npy")

            # 如果权重图已存在则跳过
            if os.path.exists(weight_path):
                continue

            # 加载标签图像
            try:
                # 直接使用txt中的完整路径
                label_img = Image.open(full_path)
                label_array = np.array(label_img)
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                continue

            # 计算权重图
            weight_map = self.compute_weight_map(label_array)

            # 保存权重图
            np.save(weight_path, weight_map)

            # 保存可视化图像
            plt.imsave(
                weight_path.replace('.npy', '.png'),
                weight_map,
                cmap="viridis",
                vmin=weight_map.min(),
                vmax=weight_map.max()
            )
            print(f"Saved weight map for {filename}")

    def compute_weight_map(self, label_array):
        """计算单个标签的权重图"""
        height, width = label_array.shape   # 512*512
        unique_classes = np.unique(label_array) # [0,1]
        num_classes = len(unique_classes)       # 2

        # 1. 计算类别权重
        class_pixels = np.zeros(num_classes)  #长度为2 统计0和1的数量
        for i, c in enumerate(unique_classes):
            class_pixels[i] = np.sum(label_array == c)

        # 添加epsilon防止除零
        total_pixels = height * width
        class_weights = total_pixels / (class_pixels + self.eps)   #数量越小的类别 权重值越大
        class_weights /= class_weights.max()  # 归一化

        # 创建类别权重映射 为每个像素分配对应的cw权重
        cw_map = np.zeros_like(label_array, dtype=np.float32)
        for i, c in enumerate(unique_classes):
            cw_map[label_array == c] = class_weights[i]

        # 2. 计算边界权重
        # 创建二值掩膜（前景=1，背景=0）处理前景有多种类型的细胞 对于二分类任务来说可以不加
        binary_mask = (label_array > 0).astype(np.uint8)
        dw_map = np.zeros((height, width), dtype=np.float32)

        if np.any(binary_mask):
            # 使用正确的连通区域标记
            '''   ndimage.label
            input:binary_mask   structure---np.ones((3, 3)) 表示使用 3x3 的邻域，即 8-连通 (8-connected)
            output: labeled 和 binary_mask尺寸一样 每个连通区域被赋予唯一的整数标签（从 1 开始） 背景保持为0
                    num_labels 检测到的连通区域总数（整数）
            '''
            labeled, num_labels = ndimage.label(
                binary_mask,
                structure=np.ones((3, 3))  # 8-连通
            )

            if num_labels >= 2:
                # 计算每个实例的距离变换
                #dist_maps (num_labels, height, width)  保存每个像素到最近实例边界的距离
                dist_maps = np.zeros((num_labels, height, width), dtype=np.float32)

                for k in range(1, num_labels + 1):
                    cell_mask = (labeled == k) #labeled中等于k的像素值为True 其他为False
                    '''
                    distance_transform_edt 函数 计算图像中非零点到最近背景点（即0）的距离 
                    '''
                    #计算实例内部到边界的距离
                    internal_dist = ndimage.distance_transform_edt(cell_mask)
                    # 计算实例外部到边界的距离
                    external_dist = ndimage.distance_transform_edt(~cell_mask)
                    # 组合: 整个图像中每个像素到该实例边界的距离
                    # 实例内部用内部距离，实例外部用外部距离
                    combined_dist = np.where(cell_mask, internal_dist, external_dist)

                    dist_maps[k - 1] = combined_dist

                # 计算每个像素到最近两个实例的距离
                #dist_maps (num_labels, height, width)
                dist_maps.sort(axis=0)  # 沿实例维度排序 从小到大
                d1 = dist_maps[0]
                d2 = dist_maps[1] if num_labels >= 2 else np.zeros_like(d1) #如果num_labels=1,防止越界

                # 计算边界权重
                dw_map = self.w0 * np.exp(-(d1 + d2) ** 2 / self.sigma_sq)

        # 3. 组合权重并确保数值稳定
        weight_map = cw_map + dw_map

        # 检查并修复NaN/Inf
        if np.isnan(weight_map).any() or np.isinf(weight_map).any():
            print("Warning: NaN/Inf detected in weight map. Replacing with 1.0")
            weight_map = np.nan_to_num(weight_map, nan=1.0, posinf=1.0, neginf=1.0)
            weight_map = np.clip(weight_map, 1e-3, 10 * self.w0)  # 安全范围

        return weight_map

    def execute(self, pred, target, targetid):
        """
        参数:
            pred: 模型预测值 (batch_size, num_classes, height, width)
            target: 真实标签 (batch_size, height, width)
            targetid: 标签文件名列表 (batch_size)
        返回:
            加权交叉熵损失
        """
        batch_size, num_classes, height, width = pred.shape

        # 裁剪logits值域 [-50, 50] 防止exp溢出
        pred = jt.clamp(pred, -50.0, 50.0)

        # 确保标签在有效范围内
        target = target.long()
        if (target.min() < 0) or (target.max() >= num_classes):
            target = jt.clamp(target, 0, num_classes - 1)

        # 加载预计算的权重图
        weight_map_batch = jt.zeros((batch_size, height, width),
                                    dtype='float32')

        for i, filename in enumerate(targetid):
            if filename in self.weights_cache:
                weight_map = np.load(self.weights_cache[filename])
                # 检查权重图异常值
                if np.isnan(weight_map).any() or np.isinf(weight_map).any():
                    print(f"Warning: Weight map {filename} contains NaN/Inf. Replacing with 1.0")
                    weight_map = np.nan_to_num(weight_map, nan=1.0, posinf=1.0, neginf=1.0)
                weight_map_batch[i] = jt.array(weight_map)
            else:
                # 如果权重图未预计算，回退到实时计算
                print(f"Warning: Weight map for {filename} not precomputed. Calculating on the fly.")
                label_array = target[i].numpy()
                weight_map = self.compute_weight_map(label_array)
                weight_map = np.clip(weight_map, 1e-3, 100)
                weight_map_batch[i] = jt.array(weight_map)

        # 添加epsilon防止log(0)
        log_probs = nn.log_softmax(pred + self.eps, dim=1) #pred (B,C,H,W) -> log_probs(B,C,H,W)

        # 安全收集目标类别的log概率
        target_expanded = target.unsqueeze(1)  # 增加通道维度 #target (B,H,W) -> target_expanded (B,1,H,W)
        '''
        jt.gather 沿维度 dim=1（类别维度），从 log_probs 中提取目标标签指定的值
        对于每个位置 (b, h, w)  根据target_expanded[b, 0, h, w]的值  0或1
        从 log_probs[b, :, h, w]中取出第 0或1 个元素
        输出形状: 与 target_expanded 相同 (B, 1, H, W)  selected_log_probs为每个像素真实类别的 对数概率值 
        '''
        selected_log_probs = jt.gather(log_probs, 1, target_expanded).squeeze(1)

        # 计算加权损失
        weighted_loss = -weight_map_batch * selected_log_probs
        mean_loss = weighted_loss.mean()


        return mean_loss


if __name__ == '__main__':
    from skimage import io
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import measure, color
    import cv2

    # 读取图像并二值化
    gt = io.imread(r'C:\Users\24174\Desktop\cmmSecond\Unet-jittor\data\ISBC2012\train\labels\frame_0001.png')
    gt = 1 * (gt > 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(gt, cmap='gray')
    plt.title('Ground Truth')
    plt.colorbar()
    plt.show()

    # 【1】计算细胞和背景的像素频率
    c_weights = np.zeros(2)
    c_weights[0] = 1.0 / ((gt == 0).sum())
    c_weights[1] = 1.0 / ((gt == 1).sum())

    # 【2】归一化
    c_weights /= c_weights.max()

    # 【3】得到 class_weight map(cw_map)
    cw_map = np.where(gt == 0, c_weights[0], c_weights[1])

    plt.figure(figsize=(10, 10))
    im = plt.imshow(cw_map, cmap='viridis')
    plt.title('Class Weight Map')
    plt.colorbar(im)
    plt.show()

    # 【4】连通域分析，并彩色化
    cells = measure.label(gt, connectivity=2)
    cells_color = color.label2rgb(cells, bg_label=0, bg_color=(0, 0, 0))

    plt.figure(figsize=(20, 20))
    plt.imshow(cells_color)
    plt.title('Labeled Cells (Color)')
    plt.axis('off')  # 彩色标注图一般不加 colorbar
    plt.show()

    # 【5】计算 distance weight map (dw_map)
    w0 = 10
    sigma = 512*0.08
    sigma_sq = 2*sigma * sigma
    dw_map = np.zeros_like(gt, dtype=float)

    if cells.max() >= 2:
        # 为每个实例计算距离变换（实例内部到边界的距离）
        dist_maps = np.zeros((cells.max(), gt.shape[0], gt.shape[1]), dtype=np.float32)

        for i in range(1, cells.max() + 1):
            cell_mask = (cells == i)

            # 方法1: 计算实例内部到边界的距离
            internal_dist = ndimage.distance_transform_edt(cell_mask)

            # 方法2: 计算实例外部到边界的距离
            external_dist = ndimage.distance_transform_edt(~cell_mask)

            # 组合: 整个图像中每个像素到该实例边界的距离
            # 实例内部用内部距离，实例外部用外部距离
            combined_dist = np.where(cell_mask, internal_dist, external_dist)

            dist_maps[i-1]=combined_dist

        # 按像素对距离图排序（升序）
        dist_maps.sort(axis=0)

        # 获取最近的两个实例的距离
        d1 = dist_maps[0]
        d2 = dist_maps[1] if cells.max() >= 2 else np.zeros_like(d1)
        # 调试信息

        dw_map = w0 * np.exp(-(d1 + d2) ** 2 / sigma_sq)

    plt.figure(figsize=(10, 10))
    im = plt.imshow(dw_map, cmap='jet')
    plt.title('Distance Weight Map')
    plt.colorbar(im)
    plt.show()

    # 最终权重图
    finalmap = cw_map + dw_map

    plt.figure(figsize=(10, 10))
    im = plt.imshow(finalmap, cmap='viridis')
    plt.title('Final Weight Map')
    plt.colorbar(im)
    plt.show()
