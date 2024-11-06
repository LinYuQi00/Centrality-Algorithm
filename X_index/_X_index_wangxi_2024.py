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

def flow_attribute_angle(OD: ArrayLike) -> Union[np.ndarray, float]:
    """calaulate flow angle.

    Args:
        OD (ArrayLike): OD matrix has 4 columns:  ox, oy, dx, dy.

    Returns:
        Union[np.ndarray, float]: angle of flow. 0-2π.
    """
    if np.ndim(OD) == 1:
        ox, oy, dx, dy = OD[0], OD[1], OD[2], OD[3]
        res =np.array([ox - dx, oy - dy])
    else:
        ox, oy, dx, dy = OD[:, 0], OD[:, 1], OD[:, 2], OD[:, 3]
        res = np.vstack((ox - dx, oy - dy)).T

    vec = res.reshape(-1, 2)
    angles = np.arctan2(vec[:, 1], vec[:, 0])
    # # 0-2π
    # theta = np.mod(angles, 2 * np.pi)
    # 将角度转换为度并转换为逆时针角度,规范到(0-360°)
    angles_deg = np.degrees(angles)
    angles_deg = (angles_deg + 360) % 360  # 将角度规范到 [0, 360)
    return angles_deg

def calculate_h_index(bin_counts):
    """根据给定的 bin_counts 计算 H 指数"""
    # 计算 H 指数
    sorted_bins = np.sort(bin_counts[bin_counts > 0])[::-1]
    return np.sum(sorted_bins >= np.arange(1,sorted_bins.size+1))

def get_bin(angle, b, alpha):
    # 调整角度，将其偏移 alpha 后限制到 [0, 360) 范围
    adjusted_angle = (angle - alpha) % 360
    # 计算分箱宽度
    bin_width = 360 / b
    # 计算分箱编号
    bin_index = int(adjusted_angle // bin_width)
    return bin_index

def calculate_x_index(flow_matrix: np.ndarray, OD: Optional[np.ndarray] = None, angles_matrix: Optional[np.ndarray] = None) -> np.ndarray:
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
    print("点流入流量",inflow_volumes)
    max_inflow = np.max(inflow_volumes)
    print(max_inflow)
    # 确定 b_max
    b_max = 1 if max_inflow == 0 else int(np.sqrt(max_inflow))
    print("b_max为：",b_max)
    # 初始化 X-index 结果数组
    x_indexes = np.zeros(flow_matrix.shape[1])  # 根据目的地数量初始化

    # 如果没有提供角度向量，且提供了 OD 矩阵，则计算角度
    if OD is not None and angles_matrix is None:
        angles_vector = flow_attribute_angle(OD)
        # 创建一个 n*m 的矩阵，n 是流量矩阵的列数，m 是流量矩阵的行数
        m, n = flow_matrix.shape[0], flow_matrix.shape[1]
        angles_matrix = np.zeros((m, n))

        # 获取流量矩阵中非零元素的位置
        non_zero_indices = np.argwhere(flow_matrix > 0)

        # 对 non_zero_indices 按列排序
        sorted_indices = non_zero_indices[np.argsort(non_zero_indices[:, 1])]

        # 按列顺序填入角度值
        angles_matrix[sorted_indices[:, 0], sorted_indices[:, 1]] = angles_vector
        # angles_matrix 现在已被正确填充

    print(angles_matrix)
    for j in range(flow_matrix.shape[1]):  # 遍历每个目的地
        # 获取当前目的地的流量信息
        flows_to_j = np.where(flow_matrix[:, j] > 0)[0]  # 找出所有流向当前目的地 j 的出发点

        if flows_to_j.size == 0:
            x_indexes[j] = 0
            continue

        # 获取当前目的地的角度，但只保留非零流量对应的角度
        angles = angles_matrix[:, j]
        print("angles:",angles)
        print("flow_matrix",flow_matrix[:,j])
        max_h_value = 0
        # 遍历从 1 到 b_max 的所有 b
        for b in range(1, b_max + 1):
            alpha_max = 360 / b  # 计算当前 b 对应的 alpha_max

            # 计算不同百分位数的 alpha
            percentiles = [0, 20, 40, 60, 80]
            for percentile in percentiles:
                alpha = (percentile / 100) * alpha_max  # 计算当前的 alpha

                # 初始化 bin_counts
                bin_counts = np.zeros(b)
                # for angle in angles:
                #     bin_index = get_bin(angle,b,alpha)
                #     if 0 <= bin_index < b:  # 确保 bin_index 在有效范围内
                #         bin_counts[bin_index] += 1
                for k, angle in enumerate(angles):
                    bin_index = get_bin(angle, b, alpha)
                    if 0 <= bin_index < b:
                        bin_counts[bin_index] += flow_matrix[k][j]  # 将流量累加到分箱
                # 计算 H(A_b, α)
                h_value = calculate_h_index(bin_counts)
                print(b, "分法,alpha_max是",alpha_max,"此时的alpha是：",alpha,"此时的百分数为:", percentile, "得到的bin分箱是：", bin_counts,",此时的h-index是：",h_value)
                max_h_value = max(max_h_value, h_value)  # 更新最大 H 值

        # 将最大 H(A_b) 值存入 X-index 数组
        x_indexes[j] = max_h_value

    return x_indexes

if __name__ == '__main__':
    flow_matrix = np.array([[1,2],
                            [1,2],
                            [1,2],
                            [1,2],
                            [1,2],
                            [1,2],
                            [1,2],
                            [1,1],
                            [1,1]])

    OD = np.array([[3,2,0,0],
                   [1,2,0,0],
                   [2,1,0,0],
                   [-2,1,0,0],
                   [-1,2,0,0],
                   [-3,2,0,0],
                   [-1,-2,0,0],
                   [0,-1,0,0],
                   [1,-2,0,0],
                   [3, 2, 1, 0],
                   [1, 2, 1, 0],
                   [2, 1, 1, 0],
                   [-2, 1, 1, 0],
                   [-1, 2, 1, 0],
                   [-3, 2, 1, 0],
                   [-1, -2, 1, 0],
                   [0, -1, 1, 0],
                   [1, -2, 1, 0]
                   ])

    x_index = calculate_x_index(flow_matrix,OD, angles_matrix= None)
    print(x_index)
