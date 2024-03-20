from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import _DistanceMatrix
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import CNN_network

##########################################展示树重建效果###########################################################
#获取给定树结构的距离矩阵，用于draw函数
def tree_toplogy_distance(path):
        #从文件中读取Newick格式的树
    tree = Phylo.read(path, "newick")
    #把树的branch长度置为1
    for clade in tree.find_clades():
        clade.branch_length = 1
    # 获取所有叶子节点的名称
    leaf_names = [leaf.name for leaf in tree.get_terminals()]
    # 初始化距离矩阵
    distances = pd.DataFrame(index=leaf_names, columns=leaf_names)
    # 计算叶子节点间的距离并填充矩阵
    for i, name1 in enumerate(leaf_names):
        for name2 in leaf_names[i + 1:]:
            distance = tree.distance(name1, name2)
            distances.at[name1, name2] = distance
            distances.at[name2, name1] = distance
    # 使用numpy来填充对角线
    np.fill_diagonal(distances.values, 0)
    return distances

#给定树文件地址，画树
def draw_tree(file):
    real_tree = Phylo.read(file, format="newick")

    for clade in real_tree.find_clades():
        clade.branch_length = 1
    for clade in real_tree.find_clades():
        if clade.is_terminal():  
            continue  
        else:
            clade.name = None
    Phylo.draw(real_tree)

def draw_UPGMA_tree(file):
    def tree_toplogy_distance(path):
        #从文件中读取Newick格式的树
            tree = Phylo.read(path, "newick")
            #把树的branch长度置为1
            for clade in tree.find_clades():
                clade.branch_length = 1
    
            # 获取所有叶子节点的名称
            leaf_names = [leaf.name for leaf in tree.get_terminals()]
            # 初始化距离矩阵
            distances = pd.DataFrame(index=leaf_names, columns=leaf_names)
    
            # 计算叶子节点间的距离并填充矩阵
            for i, name1 in enumerate(leaf_names):
                for name2 in leaf_names[i + 1:]:
                    distance = tree.distance(name1, name2)
                    distances.at[name1, name2] = distance
                    distances.at[name2, name1] = distance
    
            # 使用numpy来填充对角线
            np.fill_diagonal(distances.values, 0)
    
            return distances

    df=tree_toplogy_distance(path=file)
    print(df.index.tolist())
    tree=UPGMA(matrix=df.values.tolist(),labels=df.index.tolist())

    for clade in tree.find_clades():
        if clade.is_terminal():  
            continue  
        else:
            clade.name = None
    Phylo.draw(tree)

def draw_NJ_tree(file):
    def tree_topology_distance_nj(path):
        """
        从Newick格式的树文件中读取树的拓扑距离矩阵，并返回下三角矩阵和名称列表。

        :param path: Newick格式的树文件路径
        :return: 下三角距离矩阵和名称列表
        """
        # 从文件中读取Newick格式的树
        tree = Phylo.read(path, "newick")
        # 把树的branch长度置为1
        for clade in tree.find_clades():
            clade.branch_length = 1

        # 获取所有叶子节点的名称
        leaf_names = [leaf.name for leaf in tree.get_terminals()]
        num_leaves = len(leaf_names)

        # 初始化距离矩阵和名称列表
        distances = [[0]]  # 第一行只有一个元素
        names = [leaf_names[0]]  # 第一个叶子节点

        # 计算叶子节点间的距离并填充矩阵
        for i in range(1, num_leaves):
            distances.append([tree.distance(leaf_names[i], leaf_names[j]) for j in range(i + 1)])
            names.append(leaf_names[i])

        return distances, names
    distances,names=tree_topology_distance_nj(file)
    # 创建 DistanceMatrix 对象
    dm = DistanceMatrix(names,distances)

    constructor = DistanceTreeConstructor()
    nj_tree = constructor.nj(dm)
    for clade in nj_tree.find_clades():
        if clade.is_terminal():  
            continue  
        else:
            clade.name = None
    Phylo.draw(nj_tree)

#给定距离矩阵，画树

def UPGMA(labels,matrix):
    #输入距离矩阵，输出估计的系统发育树

    lower_triangle = [matrix[i][:i+1] for i in range(len(matrix))]
    
    # 将距离矩阵转换为 BioPython 可接受的形式
    dm = _DistanceMatrix(labels, lower_triangle)

    # 使用 SciPy 执行 UPGMA 聚类
    condensed_matrix = squareform(matrix)
    Z = linkage(condensed_matrix, method='average')

    # 使用 BioPython 构造树
    constructor = DistanceTreeConstructor()
    tree = constructor.upgma(dm)
    
    Phylo.draw(tree)

def NJ(labels,matrix):
    # 如果down_shape为真，转换为下三角矩阵
    down_tri_matrix = []
    for i in range(len(matrix)):
        down_tri_matrix.append(matrix[i][:i+1])
    # 创建 DistanceMatrix 对象
    dm = DistanceMatrix(labels,down_tri_matrix)

    constructor = DistanceTreeConstructor()
    nj_tree = constructor.nj(dm)
    for clade in nj_tree.find_clades():
        if clade.is_terminal():  
            continue  
        else:
            clade.name = None
    Phylo.draw(nj_tree)

##################################################################################################################

############################################评估回归的质量，误差分析#################################################
def plot_distribution(residuals_i_dim):
    plt.hist(residuals_i_dim, bins=50, alpha=0.5)  # 可以调整 bins 参数来改变柱状图的区间数量
    plt.xlabel(f"Residuals on Dimension {i}")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals Along Dimension {}".format(i))
    plt.show()

def get_residuals(model, data_loader,device):
    '''
    models: 输入的模型list
    dataset: 数据集包含x作为模型输入,y作为标签
    return: 模型在数据集上的残差数组(三维array)
            array[1]表示第1个模型在数据集上的残差
            array[1][2]表示模型1在数据集上第一条数据的残差
    '''
    residuals=[]
    model.to(device)
    model.eval()
    progress_bar = tqdm(data_loader, desc="Model Residuals")
    with torch.no_grad():
        # 遍历数据集中的每个样本
        for data in progress_bar:
            # 计算模型预测值与真实标签之间的残差
            input, label = data[0].to(device), data[1].to(device)
            residuals.append((model(input) - label).item())

        # 将所有模型的残差列表转换array
        residuals = np.array(residuals)

        return residuals
################################################测试部分###########################################################
if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 加载模型
    model = CNN_network.resnet_for_matrix(task="regression", encoder_type="one_hot")
    model.load_state_dict(torch.load(f'model/resnt_2_layer_model.pth'))
    
    # 加载数据
    file_path = 'data\GTR_testing_tensor\one_hot_regression_1_to_500_batch0.pt'
    testing_dataset = torch.load(file_path)
    
    #创建data_loader
    data_loader = torch.utils.data.DataLoader(
        testing_dataset,
        batch_size=100,  # 替换为你的批量大小
        shuffle=True,  # 根据需要设置是否打乱顺序
    )
    
    residuals=get_residuals(model, data_loader,device)