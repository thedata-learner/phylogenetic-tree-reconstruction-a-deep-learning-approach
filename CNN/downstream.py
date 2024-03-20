##################################################
#把模型输出转化成最终的距离矩阵
##################################################
from make_tensor import read_fasta,DataFrame2tensor
import torch
import network


###################################数据输出前处理#####################################
def DNA_to_Dataframe(tree_No,loci_num,file_path,encoder_type=1):
    '''
    tree_No:树编号
    loci_num:loci数目
    file_path:树的文件夹
    encoder_type:编码方式
    函数读取同一个树的所有分区DNA列,返回DNA_list中每一个元素代表一个分区的DNA的dataframe数据
    '''
    DNA_list=[]
    
    for i in range (loci_num):
        DNA_path=f"{file_path}/tree{tree_No}_{i}_simulated_alignment.fasta"
        DNA_list.append(read_fasta(file_path=DNA_path,encoder_type=encoder_type))

    return DNA_list

def Dataframe_to_input(DNA_list):
    '''
    DNA_list:list中的每一个元素代表一个分区的DNA

    batch:一个分区的DNA_dataframe转化成模型输入,通过调整dataframe行排序,让模型预测一共十个的距离,
           一个batch可以产生一个分区推理的距离矩阵
    
    batchlist:batch的list,表示所有分区
    '''
    idx=['Sp10','Sp1', 'Sp2', 'Sp3', 'Sp4', 'Sp5', 'Sp6', 'Sp7', 'Sp8', 'Sp9']
    batch_list=[]

    for df in DNA_list:
        batch=[]
        for i in range(10):

            first_element = idx.pop(0)
            idx.append(first_element)

            batch.append(DataFrame2tensor(df,idx).unsqueeze(0))
        
        batch = torch.stack(batch)
        batch_list.append(batch)
    
    return batch_list

def input_to_matrix(model,batch_list):

    def roll_back(list,k):
        for i in range(k):
            list.insert(0, list.pop())
        return list
    
    dis_matrix_list=[]

    for i in batch_list:
        matrix=[]

        out_tensor=model(i)
        for j in range(10):
            matrix.append(roll_back(out_tensor[j].tolist(),j))

        dis_matrix_list.append(matrix)

    return dis_matrix_list


#######################################################################################

##################################数据输出后处理#########################################
def distance_matrix_fusion(dis_matrix_list, fusion_type="AVG"):
    '''
    融合同一模型基于不同位点推出的距离矩阵估计值
    '''
    if fusion_type == "AVG":
        # 获取矩阵的维度
        rows, cols = len(dis_matrix_list[0]), len(dis_matrix_list[0][0])
        # 初始化平均矩阵
        AVG_matrix = [[0] * cols for _ in range(rows)]
        # 计算所有矩阵的平均值
        for matrix in dis_matrix_list:
            for i in range(rows):
                for j in range(cols):
                    AVG_matrix[i][j] += matrix[i][j]
        for i in range(rows):
            for j in range(cols):
                AVG_matrix[i][j] /= len(dis_matrix_list)
        return AVG_matrix
    else:
        # 默认情况下返回第一个矩阵
        return dis_matrix_list[0]

def normalized_distance_matrix(matrix, int_type=False):
    '''
    把matrix做相应的变换,使之能被基于距离矩阵的方法识别
    '''
    # 把matrix化为对称矩阵，对角线置为0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i != j:
                # 计算对称位置上的元素平均值
                avg_value = (matrix[i][j] + matrix[j][i]) / 2.0
                matrix[i][j] = avg_value
                matrix[j][i] = avg_value
            else:
                matrix[j][i] = 0
                
    # 如果int_type为真，那么对matrix所有位置元素四舍五入
    if int_type:
        matrix = [[round(elem) for elem in row] for row in matrix]

    return matrix

############################################################################

if __name__ == '__main__':
    
    # 加载模型
    model = network.resnet_for_matrix(10)
    model.load_state_dict(torch.load('model/best_model_network_restnet_01.pth'))


    #
    DNA_dataframe_list=DNA_to_Dataframe(
        tree_No=10,
        loci_num=10,
        file_path='show_data',
        encoder_type=1
    )

    batch_list=Dataframe_to_input(DNA_dataframe_list)
    matrix_list=input_to_matrix(model=model,batch_list=batch_list)

    d=distance_matrix_fusion(matrix_list)
    a=normalized_distance_matrix(matrix=matrix_list[0])
    b=normalized_distance_matrix(matrix=matrix_list[0],down_shape=True)
    c=normalized_distance_matrix(matrix=matrix_list[0],int_type=True)

    

    
    
    print("down")