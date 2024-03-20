'''
该部分内容是一个make_tensor类,用于实现把树和DNA文件转化为网络可以训练的数据并且保存
'''
import pandas as pd
import torch
from Bio import Phylo
import numpy as np
from torch.utils.data import TensorDataset
from tqdm import tqdm
import math
import psutil
import threading
###############################################################################################################
#encoder方法，输入DNA字符串，输出列表
def numeric(sequence):
    mapping = {'A': 0, 'C': 1, 'T': 2, 'G': 3}
    numeric_sequence = [mapping.get(base, base) for base in sequence]
    return numeric_sequence

def one_hot(DNA_string):
    mapping = {'A': [0, 0, 0, 1], 'C': [0, 0, 1, 0], 'T': [0, 1, 0, 0], 'G': [1, 0, 0, 0]}
    one_hot_sequence = [mapping.get(base, [0, 0, 0, 0]) for base in DNA_string]
    flattened_sequence = [bit for sublist in one_hot_sequence for bit in sublist]
    return flattened_sequence

###################################################################################################################
#DNA2Dataframe
def read_fasta(file_path, encoder_type="numeric"):
    '''
    file_path:DNA文件地址
    encoder_type:编码方式有one_hot与numeric
    return:返回dataframe结构的DNA数据,列是物种名称
    '''
    def process_sequence(encoder_function, current_species, current_sequence):
        if current_species is not None:
            species.append(current_species)
            sequences.append(encoder_function(''.join(current_sequence)))

    if encoder_type == "one_hot":
        encoder_function = one_hot
    elif encoder_type == "numeric":
        encoder_function = numeric
    else:
        raise ValueError("Invalid encoder_type: {}".format(encoder_type))

    # 用于存储物种名称和DNA序列
    species = []
    sequences = []

    # 打开fasta文件并读取内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_species = None
    current_sequence = []

    # 解析fasta文件
    for line in lines:
        if line.startswith('>'):
            # 新的物种
            process_sequence(encoder_function, current_species, current_sequence)

            current_species = line.strip()[1:]
            current_sequence = []
        else:
            # 继续当前序列
            current_sequence.append(line.strip())

    # 处理最后一个物种
    process_sequence(encoder_function, current_species, current_sequence)

    # 创建DataFrame
    df = pd.DataFrame(sequences, index=species)
    return df

#tree2Dataframe
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

#把dataframe转成tensor
def DataFrame2tensor(DataFrame, df_idx=False):
    '''
    把dataframe结构的数据转化成tensor
    输入df_idx,那么会根据df_idx的顺序对DataFrame的行进行调整
    '''
    if df_idx:
        # 如果指定了 df_idx，首先按照指定的行索引排序 DataFrame
        DataFrame = DataFrame.loc[df_idx]

    # 从 DataFrame 获取数据部分
    data = DataFrame.values
    
    # 将数据转换为 PyTorch 张量
    tensor = torch.tensor(data, dtype=torch.float32)  # 根据需要选择数据类型

    return tensor

###############################################################################################################
#为_topology_distance_classfication产生数据
def mapper(distance):
    # 创建距离的类别编码
    one_hot_matrix = torch.eye(8)
    # 选择对应距离值的行进行编码
    one_hot_encoding = one_hot_matrix[distance - 2]
    return one_hot_encoding

#为_topology_distance_regression产生数据
def get_label(distance_dataframe, idx, task):
    # 获取基准节点
    base_node = idx[0]
    # 确保基准节点存在于matrix的索引中
    if base_node not in distance_dataframe.index:
        raise ValueError(f"基准节点 '{base_node}' 不存在于DataFrame的索引中")
    
    # 获取与基准节点的距离
    if task == "classification":
        identity_matrix = np.identity(10)
        mapping_dict = { i : identity_matrix[i] for i in range(10)}
        distances = [mapping_dict[distance_dataframe.at[base_node, node]] for node in idx]

    elif task == "regression":
        distances = [distance_dataframe.at[base_node, node] for node in idx]

    #转换数据格式
    distances=torch.tensor(distances,dtype=torch.float)
    return distances

def get_order(distance_dataframe,k):
    # 读取行索引并转换为列表
    index_list = distance_dataframe.index.tolist()
    # 检查k是否在索引列表中
    if "Sp"+str(k) in index_list:
    # 移除原来的位置
        index_list.remove("Sp"+str(k))
    # 将k放到列表的第一位
        index_list.insert(0, "Sp"+str(k))
    else:
        raise ValueError(f"索引Sp '{k}' 不在 DataFrame 中")
    return index_list

def get_data(distance_dataframe, DNA_dataframe, task):
        #由distance_dataframe和DNA_dataframe生成tensor数据
        data=[]
        label=[]
        #
        species=distance_dataframe.shape[0]
        for i in range(species):
            idx=get_order(distance_dataframe, i+1)
            data.append(DataFrame2tensor(DNA_dataframe, idx).unsqueeze(0))
            label.append(get_label(distance_dataframe, idx, task))
        return data,label


######################################################################################################################
class make_tensor():
    def __init__(self, tree_flie_path, dna_file_path, save_path, 
                  loucs_num, encoder_type, task, threads_num=1):
        '''
        tree_flie_path:树的文件夹地址
        dna_file_path:DNA文件夹地址
        save_path:tensor保存地址
        loucs_num:DNA片断数目
        encoder_type:numeric or one_hot
        task:regression or classfication
        threads_num:线程数
        '''
        self.tree_flie_path = tree_flie_path
        self.dna_file_path = dna_file_path
        self.save_path = save_path
        self.loucs_num = loucs_num
        self.encoder_type = encoder_type
        self.task = task
        self.threads_num = threads_num
    
    def _topology_distance(self, start_num, end_num):
        '''
        start_num:起始树的编号
        end_num:结束树的编号
        产生数据的label是物种在系统发育树上的拓扑距离,用于回归任务
        '''
        batch=0
        data=[]
        labels=[]

        memory_threshold = 85#防止爆内存,设置当内存超过85%时自动保存数据

        for i in tqdm(range(end_num - start_num+ 1), desc="Processing Trees"):

            # 检查内存占用情况
            current_memory_usage = psutil.virtual_memory().percent
            if current_memory_usage > memory_threshold:
                # 保存已处理的数据
                data = torch.stack(data, dim=0)
                labels = torch.stack(labels, dim=0) 
                dataset = TensorDataset(data, labels)
                torch.save(dataset, f'{self.save_path}/CNN_{self.encoder_type}_{self.task}_{start_num}_to_{end_num}_batch{batch}.pt')
                
                batch=batch+1
                # 重置data和labels列表
                data = []
                labels = []

            tree_path=f'{self.tree_flie_path}/tree_{i+1}.txt'
            #读取tre e_path对应newick的树，dataframe
            distance_dataframe = tree_toplogy_distance(tree_path)

            for j in range(self.loucs_num):
            #读取DNA_path对应，转化为dataframe
                dna_path=f'{self.dna_file_path}/tree{i+1}_{j}_simulated_alignment.fasta'
                DNA_dataframe=read_fasta(dna_path,encoder_type=self.encoder_type)
                
                sub_data,sub_label=get_data(distance_dataframe, DNA_dataframe, self.task)

                data.extend(sub_data)
                labels.extend(sub_label)

        #保存tensor
        data=torch.stack(data,dim=0)
        labels=torch.stack(labels,dim=0) 
        dataset=TensorDataset(data,labels)

        torch.save(dataset, f'{self.save_path}/CNN_{self.encoder_type}_{self.task}_{start_num}_to_{end_num}_batch{batch}.pt')

###############################################################################################################
    def make_tensor(self, start_num, end_num,):
        arg_list = []
        detal = math.ceil((end_num - start_num + 1) / self.threads_num)
        for i in range(self.threads_num):
            dic={
                "start_num" : start_num + i * detal,
                "end_num" : min(start_num + (i+1) * detal - 1, end_num)
            }
            arg_list.append(dic)

        # 创建多个线程并启动
        threads = []
        for arg in arg_list:

            thread = threading.Thread(target=self._topology_distance, kwargs=(arg))
            thread.start()
            threads.append(thread)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()

############################################################################################################

def main():
    task = "testing"
    file_path = f"data/GTR_{task}_"
    loucs_num = 8
    encoder_type = "one_hot"
    task = "classification"
    threads_num = 1

    maker = make_tensor(tree_flie_path=file_path + 'tree',dna_file_path=file_path + 'DNA',
                        save_path=file_path + 'tensor', loucs_num=loucs_num, encoder_type=encoder_type,
                        task=task, threads_num=threads_num)
    
    maker._topology_distance(start_num=1, end_num=500)


if __name__ == "__main__":
    main()