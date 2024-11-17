# -*- coding: utf-8 -*-
# Author: LinYuqi
# Email: lyqllfw@gmail.com
# Date: 2024-11-13
# Description: Network Based ways for quantifying an urban city's centrality
# Papers:
# Notes:
from typing import Literal
import networkx as nx
import numpy as np

__all__ =[
    'out_degree',
    'in_degree',
    'degree_centrality',
    'weighted_degree_centrality',
    'flow_entropy',
    'recursive_centrality',
    'recursive_power',
    'eigenvector_centrality',
    'closeness_centrality',
    'betweenness_centrality'
]

def _inner_make_symmetric(matrix: np.ndarray) -> np.ndarray:
    # 创建矩阵的副本
    symmetric_matrix = matrix.copy()

    # 将下三角部分复制到上三角
    # np.tril(matrix) 获取下三角矩阵，np.tril(matrix,-1).T 获取下三角部分的转置（上三角部分,不包含主对角线）
    symmetric_matrix = np.tril(symmetric_matrix) + np.tril(symmetric_matrix, -1).T

    return symmetric_matrix

def _inner_out_degree(flow_matrix: np.ndarray) -> np.ndarray:

    out_degree = np.sum(flow_matrix > 0, axis=1)
    print(out_degree)
    return out_degree

def out_degree(flow_matrix: np.ndarray) -> np.ndarray:
    return _inner_out_degree(flow_matrix)

def _inner_in_degree(flow_matrix:np.ndarray) -> np.ndarray:
    # 计算每个节点的入度：每列中大于零的元素个数，表示其他节点向该节点流量的数量
    in_degree = np.sum(flow_matrix > 0, axis=0)

    return in_degree

def in_degree(flow_matrix:np.ndarray) -> np.ndarray:
    return _inner_in_degree(flow_matrix)

def _inner_degree_centrality(flow_matrix:np.ndarray) -> np.ndarray:
    n = flow_matrix.shape[0]  # 网络中节点的数量

    # 初始化度中心性数组
    degree_centrality = np.zeros(n)

    for i in range(n):
        # 获取出度和入度连接的节点集合
        out_nodes = set(np.where(flow_matrix[i, :] > 0)[0])  # 当前节点的出度节点集合
        in_nodes = set(np.where(flow_matrix[:, i] > 0)[0])   # 当前节点的入度节点集合

        # 判断出度和入度节点集合的关系
        if out_nodes.isdisjoint(in_nodes):
            # 如果出度和入度连接的节点集合没有交集
            degree_centrality[i] = len(out_nodes) + len(in_nodes)
        elif out_nodes.issubset(in_nodes) or in_nodes.issubset(out_nodes):
            # 如果出度和入度连接的节点集合有包含关系
            degree_centrality[i] = max(len(out_nodes), len(in_nodes))
        else:
            # 出度和入度集合有交集
            degree_centrality[i] = len(out_nodes) + len(in_nodes) - len(out_nodes.intersection(in_nodes))

    return degree_centrality

def degree_centrality(flow_matrix:np.ndarray) -> np.ndarray:
    return _inner_degree_centrality(flow_matrix)

def _inner_weighted_degree_centrality(flow_matrix: np.ndarray, mode:Literal['divergence','convergence']) -> np.ndarray:
    # 计算每个节点的加权度中心性
    if mode == 'divergence':
        centrality = np.sum(flow_matrix, axis=1)  # 沿着每一行求和
    elif mode == 'convergence':
        centrality = np.sum(flow_matrix, axis=0)
    return centrality

def weighted_degree_centrality(flow_matrix: np.ndarray, mode:Literal['divergence','convergence']) -> np.ndarray:
    return _inner_weighted_degree_centrality(flow_matrix,mode)

def _inner_flow_entropy(flows: np.ndarray, mode:Literal['divergence','convergence']) -> np.ndarray:

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
        raise ValueError("mode must be 'divergence' or 'convergence'")

    return entropies

def flow_entropy(flows: np.ndarray, mode: str) -> np.ndarray:
    return _inner_flow_entropy(flows,mode)

def _inner_recursive_centrality(R: np.ndarray) -> np.ndarray:
    # 联系矩阵R是对称矩阵，可以以此算出度中心性
    DC = _inner_degree_centrality(R)
    R = _inner_make_symmetric(R)
    RC = np.dot(R, DC)
    return RC

def recursive_centrality(R: np.ndarray) -> np.ndarray:
    return _inner_recursive_centrality(R)

def _inner_recursive_power(R: np.ndarray) -> np.ndarray:

    DC = _inner_degree_centrality(R)
    inverse_DC = 1 / DC
    R = _inner_make_symmetric(R)
    RP = np.dot(R, inverse_DC)
    return RP

def recursive_power(R: np.ndarray) -> np.ndarray:
    return _inner_recursive_power(R)

def _inner_eigenvector_centrality(flow_matrix: np.ndarray) -> np.ndarray:
    adj_matrix = (flow_matrix > 0).astype(int)
    G = nx.from_numpy_array(adj_matrix)  # 创建无向图
    eigenvector_centrality = np.array(list(nx.eigenvector_centrality(G, max_iter=1000).values()))
    return eigenvector_centrality

def eigenvector_centrality(flow_matrix: np.ndarray) -> np.ndarray:
    return _inner_eigenvector_centrality(flow_matrix)

def _inner_closeness_centrality(flow_matrix: np.ndarray) -> np.ndarray:
    adj_matrix = (flow_matrix > 0).astype(int)  # 转换为二值化邻接矩阵
    G = nx.from_numpy_array(adj_matrix)  # 创建无向图

    # 使用NetworkX的内置函数计算接近度中心性
    closeness_centrality = np.array(list(nx.closeness_centrality(G).values()))

    return closeness_centrality

def closeness_centrality(flow_matrix: np.ndarray)-> np.ndarray:
    return _inner_closeness_centrality(flow_matrix)

def _inner_betweenness_centrality(flow_matrix: np.ndarray)->np.ndarray:
    adj_matrix = (flow_matrix > 0).astype(int)
    G = nx.from_numpy_array(adj_matrix)  # 创建无向图

    # 使用NetworkX计算中介度中心性
    betweenness_centrality_dict = nx.betweenness_centrality(G, normalized=True)
    betweenness_centrality = np.array(list(betweenness_centrality_dict.values()))

    return betweenness_centrality

def betweenness_centrality(flow_matrix:np.ndarray)->np.ndarray:
    return _inner_betweenness_centrality(flow_matrix)

if __name__ == '__main__':

    # 示例：流量矩阵 (5x5的网络)
    flow_matrix = np.array([
        [0, 2, 1, 0, 0],
        [3, 0, 0, 1, 0],
        [0, 1, 0, 0, 2],
        [0, 0, 3, 0, 1],
        [2, 0, 0, 0, 0]
    ])

    # 论文中的例子
    OD_mat = np.array([
        [0, 40, 30, 20, 10],
        [80, 0, 5, 15, 0],
        [400, 300, 0, 200, 100],
        [800, 50, 150, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    mode = 'divergence'
    entropy = flow_entropy(OD_mat,mode)
    print(entropy)
    # 计算各中心性
    out_degree_centrality = _inner_out_degree(flow_matrix)
    in_degree_centrality = in_degree(flow_matrix)
    degree_centrality = degree_centrality(flow_matrix)
    recursive_centrality = recursive_centrality(flow_matrix)

    # 输出结果
    print("Out-degree Centrality:", out_degree_centrality)
    print("In-degree Centrality:", in_degree_centrality)
    print("Degree Centrality:", degree_centrality)
    print("Recursive Centrality",recursive_centrality)

