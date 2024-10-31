# -*- coding: utf-8 -*-
# Author: LinYuqi
# Email: lyqllfw@gmail.com
# Date: 2024-09-25
# Description: X-degree and MX-degree to measure the flow centrality
# paper: Wang X. et al. (2023). “A new flow-based centrality method for identifying statistically significant centers.” Sustainable Cities and Society 99, 104984.
# Note:

import numpy as np
from joblib import Parallel,delayed
from typing import Literal

def _inner_x_degree_wangxi_2024(OD_mat: np.ndarray, mode: Literal["divergence","convergence"], t: int) -> np.ndarray:
    # 检查输入的维度
    if OD_mat.ndim == 1:  # 计算单独网格的X度
        flows = OD_mat

        if mode == 'divergence' or mode == 'convergence':
            sorted_flows = np.sort(flows[flows > t])[::-1]
            return np.array([np.sum(sorted_flows >= np.arange(1, len(sorted_flows) + 1))])
        else:
            raise ValueError("mode 参数必须为 'divergence' 或 'convergence'")

    elif OD_mat.ndim == 2:  # 如果是n*n数组，则计算整个流量矩阵的X度
        n = OD_mat.shape[0]  # 矩阵的大小
        x_degree = np.zeros(n)  # 存储每个节点的 X 度
        # 根据 mode 确定遍历的方向
        for i in range(n):
            if mode == 'divergence':  # 发散模式，计算 O 点的 X 度
                flows = OD_mat[i, :]
            elif mode == 'convergence':  # 聚散模式，计算 D 点的 X 度
                flows = OD_mat[:, i]
            else:
                raise ValueError("mode 参数必须为 'divergence' 或 'convergence'")
            # 获取非零流量并按降序排序
            sorted_flows = np.sort(flows[flows > t])[::-1]
            x_degree[i] = np.sum(sorted_flows >= np.arange(1, sorted_flows.size + 1))  # 只与 sorted_flows 的大小进行比较
        return x_degree

# 封装函数，只提供给用户一个接口
def x_degree_wangxi_2024(OD_mat: np.ndarray, mode: Literal["divergence","convergence"]) -> np.ndarray:
    return _inner_x_degree_wangxi_2024(OD_mat, mode, 0)

def _inner_mx_degree_wangxi_2024(
        OD_mat: np.ndarray,
        mode: Literal["divergence","convergence"],
        threshold: np.ndarray = None
) -> np.ndarray:
    n = OD_mat.shape[0]  # 节点数

    # 初始化 MX 度数组
    mx_degree = np.zeros(n)
    # 遍历所有节点
    for i in range(n):
        node_degree = 0  # 累计该节点的 X 度
        valid_t_count = 0  # 记录有效阈值数量

        # 根据模式，确定当前节点的阈值 T
        if threshold is None:
            non_zero_values = np.sort(OD_mat[OD_mat != 0])
            # 阈值T默认值是所有流量的升序排列
        else:
            # print(threshold.ndim)
            # print(threshold.shape)
            if threshold.ndim == 1:
                non_zero_values = np.sort(threshold)  # 一维阈值数组
            elif threshold.ndim == 2 and threshold.shape[0] == n:
                non_zero_values = np.sort(threshold[i])  # 对应节点的阈值
            else:
                raise ValueError("Invalid threshold matrix dimensions.")

        # 遍历升序排列后的阈值 T，并计算 X 度
        for t in non_zero_values:
            valid_t_count += 1  # 增加阈值计数

            # 根据阈值 t 过滤流量矩阵
            if mode == 'divergence':
                filtered_OD = np.where(OD_mat[i, :] >= t, OD_mat[i, :], 0)
            elif mode == 'convergence':  # convergence 模式
                filtered_OD = np.where(OD_mat[:, i] >= t, OD_mat[i, :], 0)

            # 计算当前阈值下的 X 度
            x_degrees = x_degree_wangxi_2024(filtered_OD, mode)

            # 累加该节点的 X 度
            node_degree += x_degrees

        # 计算该节点的 MX 度：累加的 X 度 / 有效阈值数量
        if valid_t_count > 0:
            mx_degree[i] = node_degree / valid_t_count

    return mx_degree

def mx_degree_wangxi_2024(
        OD_mat: np.ndarray,
        mode: Literal["divergence","convergence"],
        threshold: np.ndarray = None
) -> np.ndarray:
    return _inner_mx_degree_wangxi_2024(OD_mat, mode, threshold)

# 封装的置换函数
def permute_labels_and_flows(OD_mat: np.ndarray) -> np.ndarray:
    n = OD_mat.shape[0]
    # 随机打乱节点标签
    permuted_labels = np.random.permutation(n)
    # 创建一个新的矩阵用于存储打乱后的流量
    permuted_mat = np.zeros_like(OD_mat)
    # 获取非零元素的位置
    non_zero_indices = np.nonzero(OD_mat)
    non_zero_positions = list(zip(non_zero_indices[0], non_zero_indices[1]))
    # 提取非零流量数据
    flows = OD_mat[non_zero_indices].copy()
    # 随机打乱流量数据
    np.random.shuffle(flows)
    # 将打乱后的流量填回原来非零的位置
    for (i, j), flow in zip(non_zero_positions, flows):
        permuted_mat[i, j] = flow
    return permuted_mat, permuted_labels


def permutation_test_mx_degree(
        OD_mat: np.ndarray,
        R: int,
        mode: Literal["divergence","convergence"],
        threshold: np.ndarray = None,
        n_jobs: int = -1
) -> np.ndarray:
    n = OD_mat.shape[0]

    # 计算原始矩阵的 MX 度
    mx_obs = mx_degree_wangxi_2024(OD_mat, mode, threshold)

    # 初始化用于存储置换后结果的计数器
    p_count = np.zeros(n)
    # 考虑使用并行计算
    permuted_results = Parallel(n_jobs=n_jobs)(
        delayed(permute_labels_and_flows)(OD_mat) for _ in range(R)
    )
    # 遍历生成的打乱结果，计算每次置换的 MX 度
    for permuted_mat, permuted_labels in permuted_results:
        mx_random = mx_degree_wangxi_2024(permuted_mat, mode, threshold)

        # 根据 permuted_labels 调整 mx_random 的顺序回到原始标签顺序
        inverse_labels = np.argsort(permuted_labels)
        mx_random_adjusted = mx_random[inverse_labels]

        # 更新 p 值的计数器
        p_count += (mx_random_adjusted >= mx_obs).astype(int)

    # 计算 p 值：p-value(j) = (Σ(Im) + 1) / (R + 1)
    p_values = (p_count + 1) / (R + 1)

    # 使用 Benjamini-Hochberg 方法调整 p 值为 q 值
    n_tests = n  # 检验次数，即节点数
    sorted_indices = np.argsort(p_values)
    q_values = np.empty_like(p_values)

    # 计算 q 值
    for rank, idx in enumerate(sorted_indices, 1):
        q_values[idx] = min(p_values[idx] * n_tests / rank, 1)

    return q_values


# test
if __name__ == "__main__":
    # 论文中的例子
    OD_mat = np.array([
        [0, 40, 30, 20, 10],
        [80, 0, 5, 15, 0],
        [400, 300, 0, 200, 100],
        [800, 50, 150, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    mode1 = 'divergence'
    mode2 = 'convergence'
    threshold = None
    threshold2 = np.array([5, 10, 15, 20, 30, 40, 50, 80, 100, 150, 200, 300, 400, 800])
    threshold3 = np.array([[5, 10, 15, 20, 30, 40, 50, 80, 100, 150, 200, 300, 400, 800],
                           [5, 10, 15, 20, 30, 40, 50, 80, 100, 150, 200, 300, 400, 800],
                           [5, 10, 15, 20, 30, 40, 50, 80, 100, 150, 200, 300, 400, 800],
                           [5, 10, 15, 20, 30, 40, 50, 80, 100, 150, 200, 300, 400, 800],
                           [5, 10, 15, 20, 30, 40, 50, 80, 100, 150, 200, 300, 400, 800]])

    R = 999

    # 发散模式（O 点的 X 度）
    result_divergence = x_degree_wangxi_2024(OD_mat, mode1)
    print("发散模式的 X 度:", result_divergence)

    # 聚集模式（D 点的 X 度）
    result_convergence = x_degree_wangxi_2024(OD_mat, mode2)
    print("聚集模式的 X 度:", result_convergence)

    result_mx_divergence = mx_degree_wangxi_2024(OD_mat, mode1, threshold)
    print("发散模式的 MX 度：", result_mx_divergence)

    result_mx_convergence = mx_degree_wangxi_2024(OD_mat, mode2, threshold)
    print("聚集模式的 MX 度:", result_mx_convergence)

    # 执行置换测试
    q_values_divergence = permutation_test_mx_degree(OD_mat, R, mode1, threshold)
    print("发散模式下校正后的 q 值:", q_values_divergence)

    q_values_convergence = permutation_test_mx_degree(OD_mat, R, mode2, threshold)
    print("聚集模式下校正后的 q 值:", q_values_convergence)
