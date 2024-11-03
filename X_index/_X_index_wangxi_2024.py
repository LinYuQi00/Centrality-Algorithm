# -*- coding: utf-8 -*-
# Author: LinYuqi
# Email: lyqllfw@gmail.com
# Date: 2024-11-1
# Description: X-index ——— flow-based locational measure for quantifying centrality
# paper: Wang X. et al.(2023). "X-index: A novel flow-based locational measure for quantifying centrality" International Journal of Applied Earth Observations and Geoinformation 117 (2023) 103187
# Note:

import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Optional
from sklearn.metrics.pairwise import cosine_similarity

def flow_attribute_OD2Vec(OD: ArrayLike) -> np.ndarray:
    """convert flow to vector, shape: N x 4 -> N x 2.

    Args:
        OD (ArrayLike): OD matrix has 4 columns:  ox, oy, dx, dy.

    Returns:
        np.ndarray: vector of flow.
    """
    if np.ndim(OD) == 1:
        ox, oy, dx, dy = OD[0], OD[1], OD[2], OD[3]
        return np.array([dx - ox, dy - oy])
    else:
        ox, oy, dx, dy = OD[:, 0], OD[:, 1], OD[:, 2], OD[:, 3]
    res = np.vstack((dx - ox, dy - oy)).T
    return res

def flow_attribute_angle(OD: ArrayLike) -> Union[np.ndarray, float]:
    """calaulate flow angle.

    Args:
        OD (ArrayLike): OD matrix has 4 columns:  ox, oy, dx, dy.

    Returns:
        Union[np.ndarray, float]: angle of flow. 0-2π.

    >>> flow = np.array([[0, 0, 1, 0],[0, 0, 1, 1],[0, 0, 0, 1],[0, 0, -1, 1],[0, 0, -1, 0],[0, 0, -1, -1],[0, 0, 0, -1],[0, 0, 1, -1],], dtype=int)
    >>> ang = flow_attribute_angle(OD=flow)
    >>> res = ang == np.array([0, np.pi * 0.25, np.pi * 0.5, np.pi * 0.75, np.pi, np.pi * 1.25, np.pi * 1.5, np.pi * 1.75])
    >>> np.all(res)
    True
    """
    vec = flow_attribute_OD2Vec(OD).reshape(-1, 2)
    angles = np.arctan2(vec[:, 1], vec[:, 0])
    # 0-2π
    theta = np.mod(angles, 2 * np.pi)
    return theta

def calculate_x_index(flow_matrix: np.ndarray, od_matrix: Optional[np.ndarray] = None, angles_vector: Optional[np.ndarray] = None) -> np.ndarray:
    """
    使用位移向量间的角度计算所有地点的 X-index。

    参数:
    - flow_matrix: 2D NumPy array, OD流量矩阵，其中每个元素表示从一个地点流向另一个地点的流量
    - od_matrix: Optional, 2D NumPy array, 每行表示一条流的 (ox, oy, dx, dy) 坐标
    - angles_vector: Optional, 1D NumPy array, 角度值

    返回:
    - x_indexes: 1D NumPy array, 每个地点的 X-index
    """
    # 计算每个地点的流入流量
    inflow_volumes = np.sum(flow_matrix, axis=0)
    max_inflow = np.max(inflow_volumes)

    # 确定 b_max
    b_max = 1 if max_inflow == 0 else int(np.sqrt(max_inflow))

    # 初始化 X-index 结果数组
    x_indexes = np.zeros(flow_matrix.shape[1])  # 根据目的地数量初始化

    # 如果没有提供角度向量，且提供了 OD 矩阵，则计算角度
    if od_matrix is not None and angles_vector is None:
        angles_vector = flow_attribute_angle(od_matrix)

    # 创建一个 n*m 的矩阵，n 是流量矩阵的列数，m 是流量矩阵的行数
    n, m = flow_matrix.shape[1], flow_matrix.shape[0]
    angles_matrix_transposed = np.zeros((n, m))

    # 获取流量矩阵中非零元素的位置
    non_zero_indices = np.argwhere(flow_matrix > 0)

    # 将角度值按顺序填入新矩阵
    for idx, (i, j) in enumerate(non_zero_indices):
        angles_matrix_transposed[j, i] = angles_vector[idx]

    # 转置得到最终的角度矩阵
    angles_matrix = angles_matrix_transposed.T

    # 遍历每个目的地，计算其 X-index
    for j in range(flow_matrix.shape[1]):  # 遍历每个目的地
        # 获取当前目的地的流量信息
        flows_to_j = np.where(flow_matrix[:, j] > 0)[0]  # 找出所有流向当前目的地 j 的出发点

        if flows_to_j.size == 0:
            x_indexes[j] = 0
            continue

        # 获取当前目的地的角度
        angles = angles_matrix[:, j]  # 使用生成的 angles_matrix

        # 将角度值分配到 b 个箱
        max_h_value = 0
        for b in range(1, b_max + 1):
            bin_counts = np.zeros(b)
            for angle in angles:
                if angle < np.pi:  # 使用一个固定的阈值过滤
                    bin_index = int((angle / np.pi) * b)
                    if 0 <= bin_index < b:  # 确保 bin_index 在有效范围内
                        bin_counts[bin_index] += 1

            # 计算 H(A_b)
            h_value = min(np.sum(bin_counts >= h) for h in range(1, b + 1))
            max_h_value = max(max_h_value, h_value)

        # 将最大 H(A_b) 值存入 X-index 数组
        x_indexes[j] = max_h_value

    return x_indexes

if __name__ == '__main__':
    print(f"hello,mytest")
