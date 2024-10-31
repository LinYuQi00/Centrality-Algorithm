# -*- coding: utf-8 -*-
# Author: LinYuqi
# Email: lyqllgw@gmail.com
# Date: 2024-10-13
# Description: X-degree and MX-degree to measure the flow centrality
# paper: Wang X. et al. (2023). “A new flow-based centrality method for identifying statistically significant centers.” Sustainable Cities and Society 99, 104984.
# Note:

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson, norm, uniform, kendalltau
from _mx_degree_wangxi_2024 import *
import jenkspy

# ============ 地理流网络生成 ============
def generate_synthetic_data( n : int , lambda_k : int , distribution : str , mode : str):
    # 初始化流量矩阵
    flows = np.zeros((n, n))

    # 生成泊松分布的连接数量，每个网格的连接数
    connection_counts = poisson.rvs(lambda_k, size=n)

    # 计算总流数（所有节点的连接数之和）
    total_flows = np.sum(connection_counts)

    # 计算 95% 置信区间的上限，识别极端网格
    threshold = poisson.ppf(0.95, lambda_k)
    extreme_indices = np.where(connection_counts > threshold)[0]

    # 根据选择的分布类型，生成 DL 和 DH 流量值，总数为 total_flows
    if distribution == 'normal_wide':
        DL_values = norm.rvs(300, 100, size=total_flows)
        DH_values = norm.rvs(800, 50, size=total_flows)
    elif distribution == 'normal_narrow':
        DL_values = norm.rvs(200, 20, size=total_flows)
        DH_values = norm.rvs(500, 10, size=total_flows)
    elif distribution == 'uniform':
        DL_values = uniform.rvs(0, 300, size=total_flows)
        DH_values = uniform.rvs(600, 10, size=total_flows)
    elif distribution == 'poisson':
        DL_values = poisson.rvs(300, size=total_flows)
        DH_values = poisson.rvs(800, size=total_flows)
    else:
        raise ValueError("未知分布类型。支持 'normal_wide', 'normal_narrow', 'uniform', 'poisson'")

    # 遍历 DL 和 DH 流量，对极端网格取最大流量值
    flow_index = 0  # 索引遍历流量数组
    for i in range(n):
        num_connections = connection_counts[i]
        if i in extreme_indices:
            # 对极端节点，取 DL 和 DH 中的较大值
            for j in range(num_connections):
                DL_values[flow_index + j] = max(DL_values[flow_index + j], DH_values[flow_index + j])
        flow_index += num_connections  # 更新流量索引

    # 填充流量矩阵
    flow_index = 0  # 重置流量索引
    for i in range(n):
        if connection_counts[i] > 0:
            # 随机选择与节点 i 连接的其他节点（确保不填入对角线）
            connected_nodes = np.random.choice(
                [j for j in range(n) if j != i], connection_counts[i], replace=False
            )

            # 将 DL_values 中的流量填入矩阵
            for j in connected_nodes:
                if mode == 'divergence':
                    flows[i, j] = DL_values[flow_index]  # 发散模式：行方向流量
                elif mode == 'convergence':
                    flows[j, i] = DL_values[flow_index]  # 聚集模式：列方向流量
                flow_index += 1

    # 确保对角线为 0（没有自流）
    np.fill_diagonal(flows, 0)

    return flows, extreme_indices

# ============ 回归分析 ============
def linear_regression_analysis(flows : np.ndarray, mx_degree : np.ndarray, mode: str):
    """
    使用多元线性回归分析总流量和流量熵对 MX-度的影响。

    参数:
        flows (np.ndarray): 形状为 (n, n) 的流量矩阵。
        mx_degree (np.ndarray): 形状为 (n,) 的 MX 度值。
        mode (str): 'divergence' 表示发散模式，'convergence' 表示聚集模式。

    返回:
        dict: 回归模型结果，包括系数和 R^2。
    """
    n = flows.shape[0]

    # 根据模式计算每个节点的总流量
    if mode == 'divergence':
        total_flows = np.sum(flows, axis=1)  # 行求和
    elif mode == 'convergence':
        total_flows = np.sum(flows, axis=0)  # 列求和
    else:
        raise ValueError("mode 参数必须为 'divergence' 或 'convergence'")

    # 计算流量熵
    entropies = flow_entropy(flows, mode)

    # 标准化所有变量
    scaler = StandardScaler()
    X = scaler.fit_transform(np.column_stack((total_flows, entropies)))
    y = scaler.fit_transform(mx_degree.reshape(-1, 1)).flatten()

    # 建立多元线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 输出回归结果
    results = {
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "R^2": model.score(X, y)
    }
    return results

# ============ 分级敏感性测试 ============
# problems unsolved
def classify_data(OD_mat: np.ndarray, method: str, n_bins: int, mode: str) -> np.ndarray:
    """根据指定方法进行数据分类（等区间、量化、自然断裂）。"""
    n = OD_mat.shape[0]
    categorized_matrix = np.zeros_like(OD_mat)  # 用于存储分类后的矩阵

    if mode == 'divergence':
        for i in range(n):  # 遍历每一行
            row_data = OD_mat[i, :]  # 获取当前行的数据
            if method == 'equal_interval':
                bins = np.linspace(np.min(row_data), np.max(row_data), n_bins + 1)
                categorized_row = np.digitize(row_data, bins)
            elif method == 'quantile':
                # 使用自定义标签 1 到 n_bins
                categorized_row = np.array(
                    pd.qcut(row_data, n_bins, labels=np.arange(1, n_bins + 1), duplicates='drop'))
            elif method == 'natural_breaks':
                # 使用 jenkspy 获取自然间断法的断点
                breaks = jenkspy.jenks_breaks(row_data, n_bins)
                categorized_row = np.digitize(row_data, bins=breaks)
            else:
                raise ValueError("Unsupported classification method")
            categorized_matrix[i, :] = categorized_row  # 保存分类结果

    elif mode == 'convergence':
        for j in range(n):  # 遍历每一列
            col_data = OD_mat[:, j]  # 获取当前列的数据
            if method == 'equal_interval':
                bins = np.linspace(np.min(col_data), np.max(col_data), n_bins + 1)
                categorized_col = np.digitize(col_data, bins)
            elif method == 'quantile':
                categorized_col = np.array(pd.qcut(col_data, n_bins, labels=False, duplicates='drop'))
            elif method == 'natural_breaks':
                # 使用 jenkspy 获取自然间断法的断点
                breaks = jenkspy.jenks_breaks(col_data, n_bins)
                categorized_col = np.digitize(col_data, bins=breaks)
            else:
                raise ValueError("Unsupported classification method")
            categorized_matrix[:, j] = categorized_col  # 保存分类结果

    return categorized_matrix


def sensitivity_analysis(OD_mat: np.ndarray, n_bins_range: range, methods: list, mode: str):
    """进行敏感度分析，计算不同分类方法和等级数下的 MX 度及其相关性。"""

    results = []
    Origin_MX_degree=mx_degree_wangxi_2024(OD_mat,mode)

    for method in methods:
        for n_bins in n_bins_range:
            categorized_matrix = classify_data(OD_mat, method, n_bins, mode)
            new_mx_degree = mx_degree_wangxi_2024(categorized_matrix, mode)

            # 计算 Kendall Tau 相关系数
            tau, p_value = kendalltau(Origin_MX_degree, new_mx_degree)
            results.append((method, n_bins, tau, p_value))

            print(f"方法: {method}, 等级数: {n_bins}, Kendall Tau: {tau:.4f}")

    return results


def flow_entropy(flows: np.ndarray, mode: str ) -> np.ndarray:
    """
    计算流量熵，根据模式不同（发散或聚集）计算。

    参数:
        flows (np.ndarray): 形状为 (n, n) 的流量矩阵。
        mode (str): 'divergence' 表示发散模式，'convergence' 表示聚集模式。

    返回:
        np.ndarray: 形状为 (n,) 的流量熵数组。
    """
    n = flows.shape[0]
    entropies = np.zeros(n)

    if mode == 'divergence':
        # 发散模式：逐行计算流量熵
        for i in range(n):
            total_flow = np.sum(flows[i, :])
            if total_flow == 0:
                entropies[i] = 0
                continue

            probabilities = flows[i, :] / total_flow
            probabilities = probabilities[probabilities > 0]
            entropies[i] = -np.sum(probabilities * np.log(probabilities))

    elif mode == 'convergence':
        # 聚集模式：逐列计算流量熵
        for j in range(n):
            total_flow = np.sum(flows[:, j])
            if total_flow == 0:
                entropies[j] = 0
                continue

            probabilities = flows[:, j] / total_flow
            probabilities = probabilities[probabilities > 0]
            entropies[j] = -np.sum(probabilities * np.log(probabilities))

    else:
        raise ValueError("mode 参数必须为 'divergence' 或 'convergence'")

    return entropies
# ============ 主程序 ============
if __name__ == "__main__":
    #  测试参数设定
    n = 100  # 流量矩阵的大小
    lambda_k = 40
    distribution = 'normal_wide'
    mode = 'divergence'
    R = 999  # 置换检验次数

    #  生成地理流矩阵
    flows, extreme_indices = generate_synthetic_data(
        n=n, lambda_k=lambda_k, distribution=distribution, mode=mode
    )

    # 打印极端网格
    print(f"极端网格: {extreme_indices}")

    # 计算原始的 MX-degree
    mx_degree = mx_degree_wangxi_2024(flows, mode=mode)
    print(f"MX-degree 计算完成")

    # 进行置换检验
    q_values = permutation_test_mx_degree(flows, R, mode=mode)
    significant_centers = np.where(q_values < 0.05)[0]
    print(f"显著中心: {significant_centers}")

    #  回归分析
    results = linear_regression_analysis(flows, mx_degree, mode=mode)

    # 打印回归分析结果
    print("\n线性回归分析结果：")
    print(f"回归系数: {results['coefficients']}")
    # print(f"截距: {results['intercept']}")
    print(f"R²: {results['R^2']:.4f}")

    # 敏感度测试
    # 设置等级
    methods = ['equal_interval', 'quantile', 'natural_breaks']
    n_bins_list = range(2, 11)  # 从 2 到 10 的等级

    sensitivity_analysis(flows,n_bins_list,methods,mode ='divergence')




