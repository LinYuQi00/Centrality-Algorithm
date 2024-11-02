# -*- coding: utf-8 -*-
# Author: LinYuqi
# Email: lyqllfw@gmail.com
# Date: 2024-11-1
# Description: X-index ——— flow-based locational measure for quantifying centrality
# paper: Wang X. et al.(2023). "X-index: A novel flow-based locational measure for quantifying centrality" International Journal of Applied Earth Observations and Geoinformation 117 (2023) 103187
# Note:

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def _inner_calculate_angles(OD: np.ndarray) -> np.ndarray:
    """
    计算位移向量之间的角度。

    参数:
    - OD (np.ndarray): 每行表示一个 OD 流的数组，格式为 (ox, oy, dx, dy)

    返回:
    - np.ndarray: 位移向量间的角度矩阵
    """
    # 计算位移向量
    vectors = OD[:, 2:4] - OD[:, 0:2]

    # 计算所有向量对之间的余弦相似度
    cos_sim = cosine_similarity(vectors)

    # 使用 arccos 计算角度（弧度）矩阵
    angles_matrix = np.arccos(np.clip(cos_sim, -1.0, 1.0))

    return angles_matrix

def _inner_x_index_wangxi_2024(flow_matrix: np.ndarray, OD_matrix: np.ndarray) -> np.ndarray:
    """
    计算每个 D 点的 X-index。

    参数:
    - flow_matrix: 2D NumPy array, OD流量矩阵
    - OD_matrix: 2D NumPy array, 每行表示一个 OD 流的数组，格式为 (ox, oy, dx, dy)

    返回:
    - x_indexes: 1D NumPy array, 每个 D 点的 X-index
    """
    # 计算每个地点的流入流量
    inflow_volumes = np.sum(flow_matrix, axis=1)  # 行求和表示流入流量
    max_inflow = np.max(inflow_volumes)

    # 确定 b_max，考虑没有流入的情况
    b_max = 1 if max_inflow == 0 else int(np.sqrt(max_inflow))  # 设置 b_max

    x_indexes = np.zeros(flow_matrix.shape[1])  # 初始化 X-index 结果数组，大小为 D 点的数量

    # 遍历每个 D 点，计算其 X-index
    for i in range(flow_matrix.shape[1]):  # 遍历 D 点
        location = OD_matrix[i + flow_matrix.shape[0]]  # 获取 D 点坐标 ????

        # 构建 OD flows 表示从所有 O 点流向 D 点 i
        flows_to_i = []
        for j in range(flow_matrix.shape[0]):  # 遍历所有 O 点
            if flow_matrix[j, i] > 0:  # 检查是否有流量
                flows_to_i.append([
                    OD_matrix[j, 0], OD_matrix[j, 1], location[0], location[1]
                ])

        flows_to_i = np.array(flows_to_i)  # 转换为 NumPy 数组

        if flows_to_i.size == 0:
            x_indexes[i] = 0
            continue

        # 计算角度矩阵
        angles_matrix = _inner_calculate_angles(flows_to_i)

        # 初始化最大 H 值
        max_h_index = 0

        # 计算 H(A_b, α) 对于每个 b 和 α 值
        for b in range(1, b_max + 1):
            alpha_max = 360 / b  # 计算 alpha 的最大值
            alpha_values = [alpha_max * k / 100 for k in [0, 20, 40, 60, 80]]  # 0%, 20%, 40%, 60%, 80%

            # 初始化每个 b 的最大 H 值
            max_h_value_for_b = 0

            for alpha in alpha_values:
                bin_counts = np.zeros(b)
                for angle in angles_matrix.flatten():
                    bin_index = int((angle / np.pi) * b)
                    if angle < np.deg2rad(alpha):  # 使用角度的弧度表示
                        bin_counts[bin_index] += 1

                # 计算 H(A_b, α) - 找到覆盖的箱数和最小流量数
                h_value = min(np.sum(bin_counts >= h) for h in range(1, b + 1))
                max_h_value_for_b = max(max_h_value_for_b, h_value)

            # 更新当前 D 点的最大 H 值
            max_h_index = max(max_h_index, max_h_value_for_b)

        # 将最大 H 值存入 X-index 数组
        x_indexes[i] = max_h_index

    return x_indexes



if __name__ == '__main__':
    print(f"hello,mytest")
