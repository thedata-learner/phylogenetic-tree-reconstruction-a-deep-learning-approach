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
from torch_geometric.data import Data

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
#产生数据
def get_data_GNN(distance_dataframe, DNA_dataframe, task):
    idx = ['Sp1', 'Sp2', 'Sp3', 'Sp4', 'Sp5', 'Sp6', 'Sp7', 'Sp8', 'Sp9', 'Sp10']
    x = torch.tensor(DNA_dataframe.reindex(idx).values, dtype=torch.float32)
    edge_attr=[]

    if task == "regression":
        for i in range(1,10):
            for j in range(i+1,11):
                edge_attr.append([distance_dataframe[f"Sp{i}"][f"Sp{j}"]])
    
    elif task == "classfication":
        I = np.identity(8)
        for i in range(1,10):
            for j in range(i+1,11):
                    edge_attr.append(I[distance_dataframe[f"Sp{i}"][f"Sp{j}"] - 2])

    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    return x, edge_attr

def get_data_TS(distance_dataframe, DNA_dataframe):
    idx = ['Sp1', 'Sp2', 'Sp3', 'Sp4', 'Sp5', 'Sp6', 'Sp7', 'Sp8', 'Sp9', 'Sp10']

    x = torch.tensor(DNA_dataframe.reindex(idx).values, dtype=torch.float32).unsqueeze(dim=0)
    y = torch.tensor(distance_dataframe.reindex(index=idx, columns=idx).values, dtype=torch.float32).unsqueeze(dim=0)

    return x,y
######################################################################################################################
class make_tensor_GNN():
    def __init__(self, tree_flie_path, dna_file_path, save_path, 
                  loucs_num, encoder_type, task,):
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
    
    def make_tensor(self, start_num, end_num,):
        '''
        start_num:起始树的编号
        end_num:结束树的编号
        产生数据的label是物种在系统发育树上的拓扑距离,用于回归任务
        '''
        batch=0
        data=[]
        edge_index = torch.combinations(torch.arange(10), r=2).t().contiguous()#全连接图

        memory_threshold = 85#防止爆内存,设置当内存超过85%时自动保存数据
        for i in tqdm(range(end_num - start_num+ 1), desc="Processing Trees"):

            # 检查内存占用情况
            current_memory_usage = psutil.virtual_memory().percent
            if current_memory_usage > memory_threshold:
                # 保存已处理的数据
                torch.save(data, f'{self.save_path}/GNN_{self.encoder_type}_{self.task}_{start_num}_to_{end_num}_batch{batch}.pt')
                batch=batch+1
                # 重置data和labels列表
                data = []
                

            tree_path=f'{self.tree_flie_path}/tree_{i+1}.txt'
            #读取tre e_path对应newick的树，dataframe
            distance_dataframe = tree_toplogy_distance(tree_path, )

            for j in range(self.loucs_num):
            #读取DNA_path对应，转化为dataframe
                dna_path = f'{self.dna_file_path}/tree{i+1}_{j}_simulated_alignment.fasta'
                DNA_dataframe = read_fasta(dna_path,encoder_type=self.encoder_type)
                
                x,edge_attr = get_data_GNN(distance_dataframe, DNA_dataframe, self.task)

                data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        
        #保存tensor
        torch.save(data, f'{self.save_path}/GNN_{self.encoder_type}_{self.task}_{start_num}_to_{end_num}_batch{batch}.pt')

class make_tensor_transformer():
    def __init__(self, tree_flie_path, dna_file_path, save_path, 
                  loucs_num, encoder_type, task,):
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
    def make_tensor(self, start_num, end_num,):
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
                torch.save(data, f'{self.save_path}/transformer_{self.encoder_type}_{self.task}_{start_num}_to_{end_num}_batch{batch}.pt')
                batch=batch+1
                # 重置data和labels列表
                data = []
                labels=[]
            tree_path=f'{self.tree_flie_path}/tree_{i+1}.txt'
            
            #读取tre e_path对应newick的树，dataframe
            distance_dataframe = tree_toplogy_distance(tree_path, )
            distance_dataframe = distance_dataframe.astype(np.float32)
            for j in range(self.loucs_num):
            #读取DNA_path对应，转化为dataframe
                dna_path = f'{self.dna_file_path}/tree{i+1}_{j}_simulated_alignment.fasta'
                DNA_dataframe = read_fasta(dna_path,encoder_type=self.encoder_type)
                x,y = get_data_TS(distance_dataframe, DNA_dataframe)

                data.extend(x)
                labels.extend(y)
                
        #保存tensor
        data = torch.stack(data,dim=0)
        labels = torch.stack(labels,dim=0) 
        dataset=TensorDataset(data,labels)

        torch.save(dataset, f'{self.save_path}/transformer_{self.encoder_type}_{self.task}_{start_num}_to_{end_num}_batch{batch}.pt')
    

############################################################################################################

def main():
    task = "val"
    tree_file_path = f"data/GTR_{task}_"+'tree'
    dna_file_path = f'data/GTR_{task}_'+'DNA'
    save_path = f'data/GTR_{task}_tensor'
    loucs_num = 10
    encoder_type = "one_hot"
    task='regression'

    maker = make_tensor_transformer(tree_flie_path=tree_file_path,
                        dna_file_path=dna_file_path,
                        save_path=save_path, 
                        loucs_num=loucs_num, 
                        encoder_type=encoder_type,
                        task=task,
                        )
    
    maker.make_tensor(start_num=1, end_num=500)


if __name__ == "__main__":
    main()




    