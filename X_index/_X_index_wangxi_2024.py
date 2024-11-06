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

__all__ = [
    "flow_centrality_x_index_wangxi_2023"
]

def _inner_flow_attribute_angle(OD: ArrayLike) -> Union[np.ndarray, float]:
    """
    Calculate the anti-clockwise angle of the OD flow with respect to the due east direction

    Args:
        OD (ArrayLike): OD matrix has 4 columns:  ox, oy, dx, dy.

    Returns:
        Union[np.ndarray, float]: angle of flow. 0-360°
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

def _inner_calculate_h_index(bin_counts: np.array)-> int:
    """
    Calculate the H-index based on given bin counts.

    The H-index quantifies the concentration of flow across bins. It is computed by sorting the bin
    counts in descending order and finding the largest `k` such that there are at least `k` bins with
    a count greater than or equal to `k`.

    Args:
        bin_counts (np.ndarray): A 1D array of flow counts for each angle bin.

    Returns:
        int: The calculated H-index.
    """
    # 计算 H 指数
    sorted_bins = np.sort(bin_counts[bin_counts > 0])[::-1]
    return np.sum(sorted_bins >= np.arange(1,sorted_bins.size+1))

def _inner_get_bin(angle: float, b: int, alpha: float)-> int:
    """
    Calculate the bin index for a given angle, with an adjustment by alpha.

    The function adjusts the angle by the offset `alpha`, normalizes it to the range [0, 360),
    then determines the corresponding bin index based on the number of bins `b`. The angle is
    divided into `b` bins, each of equal width, and the bin index is calculated by dividing the
    adjusted angle by the bin width.

    Args:
        angle (float): The angle to be assigned to a bin, in degrees (0-360).
        b (int): The number of bins to divide the angle range into.
        alpha (float): The angle offset to adjust the input angle before binning.

    Returns:
        int: The index of the bin corresponding to the given angle.
    """
    # 调整角度，将其偏移 alpha 后限制到 [0, 360) 范围
    adjusted_angle = (angle - alpha) % 360
    # 计算分箱宽度
    bin_width = 360 / b
    # 计算分箱编号
    bin_index = int(adjusted_angle // bin_width)
    return bin_index

def _inner_x_index_wangxi_2023(flow_matrix: np.ndarray, OD: Optional[np.ndarray] = None, angles_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the X-index for flow centrality using the flow matrix and optional OD trip matrix or angles matrix.

    Paper Reference: Wang X. et al.(2023). "X-index: A novel flow-based locational measure for quantifying centrality" International Journal of Applied Earth Observations and Geoinformation 117 (2023) 103187

    Args:
        flow_matrix (np.ndarray): A 2D NumPy array representing the origin-destination (OD) flow matrix (m x n).
        OD (Optional[np.ndarray], optional): A 2D NumPy array (m x 4) containing the coordinates of each flow (origin_x, origin_y, destination_x, destination_y).
        angles_matrix (Optional[np.ndarray], optional): A 2D NumPy array (m x n) containing pre-calculated flow angles for each (origin, destination) pair.

    Returns:
        np.ndarray: A 1D NumPy array containing the X-index values for each destination node in the flow network.Higher values indicate more central locations in the network based on flow dynamics.
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
        angles_vector = _inner_flow_attribute_angle(OD)
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
                    bin_index = _inner_get_bin(angle, b, alpha)
                    if 0 <= bin_index < b:
                        bin_counts[bin_index] += flow_matrix[k][j]  # 将流量累加到分箱
                # 计算 H(A_b, α)
                h_value = _inner_calculate_h_index(bin_counts)
                print(b, "分法,alpha_max是",alpha_max,"此时的alpha是：",alpha,"此时的百分数为:", percentile, "得到的bin分箱是：", bin_counts,",此时的h-index是：",h_value)
                max_h_value = max(max_h_value, h_value)  # 更新最大 H 值

        # 将最大 H(A_b) 值存入 X-index 数组
        x_indexes[j] = max_h_value

    return x_indexes

def flow_centrality_x_index_wangxi_2023(flow_matrix: np.ndarray, OD: Optional[np.ndarray] = None, angles_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the X-index for flow centrality using the flow matrix and optional OD trip matrix or angles matrix.

    This function calculates the X-index for each destination in a flow network, which measures the
    centrality of a location based on the distribution of incoming flows. The X-index incorporates both
    the volume and directionality of flows toward a destination, quantifying its centrality relative to
    other nodes in the network.

    Paper Reference: Wang X. et al.(2023). "X-index: A novel flow-based locational measure for quantifying centrality" International Journal of Applied Earth Observations and Geoinformation 117 (2023) 103187

    Args:
        flow_matrix (np.ndarray): A 2D NumPy array representing the origin-destination (OD) flow matrix (m x n).
        OD (Optional[np.ndarray], optional): A 2D NumPy array (m x 4) containing the coordinates of each flow (origin_x, origin_y, destination_x, destination_y).
        angles_matrix (Optional[np.ndarray], optional): A 2D NumPy array (m x n) containing pre-calculated flow angles for each (origin, destination) pair.

    Returns:
        np.ndarray: A 1D NumPy array containing the X-index values for each destination node in the flow network.Higher values indicate more central locations in the network based on flow dynamics.
    """
    return _inner_x_index_wangxi_2023(flow_matrix,OD,angles_matrix)

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

    x_index = flow_centrality_x_index_wangxi_2023(flow_matrix, OD, angles_matrix= None)
    print(x_index)
