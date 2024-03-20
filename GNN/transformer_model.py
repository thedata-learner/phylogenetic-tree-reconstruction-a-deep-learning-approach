import torch
import torch.nn as nn
##################################################################################################################
class encoder_block(nn.Module):
    def __init__(self, 
                 input_dim, 
                 #output_dim
                 ) -> None:
        
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=10, dim_feedforward=2048, dropout=0.2, batch_first=True)
        #self.ffn = nn.Linear(input_dim, output_dim)
        #self.relu = nn.ReLU()

    def forward(self, x):
        x = self.transformer(x)
        #x = self.ffn(x)
        #x = self.relu(x)
        return x
##################################################################################################################
class resnet_block(nn.Module):
    def __init__(self,) -> None:
        super().__init__()

    def forward(self, x):

        return x
#######################################################################################################################
class output_block1(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(output_block1, self).__init__()
        # 定义前馈神经网络，输入维度是两倍input_dim
        self.ffn = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_matrix):
        # 获取输入矩阵的大小
        batch_size, _, input_dim = input_matrix.shape
        
        # 初始化输出矩阵并将其对角线元素初始化为0
        output_matrix = torch.zeros((batch_size, 10, 10), device=input_matrix.device)

        # 遍历batch中的每一个样本
        for b in range(batch_size):
            # 对于当前batch的样本
            sample = input_matrix[b]

            # 计算所有非对角线元素
            for i in range(10):
                for j in range(i + 1, 10):  # 只遍历非对角线元素
                    # 拼接输入矩阵的i行和j行
                    concatenated_vector = torch.cat([sample[i], sample[j]], dim=-1)

                    # 通过前馈神经网络计算单个值
                    off_diagonal_element = self.ffn(concatenated_vector).squeeze(-1)

                    # 将计算出的值赋给输出矩阵的非对角线位置
                    output_matrix[b, i, j] = off_diagonal_element
                    output_matrix[b, j, i] = off_diagonal_element  # 因为矩阵是对称的

        return output_matrix

class output_block2(nn.Module):
    # 输入维度 (batchsize, 10, input_dim)，输出维度 (batchsize, 10)
    def __init__(self, input_dim, hidden_dim=64):
        super(output_block2, self).__init__()
        
        # 定义前馈神经网络，将input_dim维的向量映射到hidden_dim维，然后再映射回10维
        self.ffn = nn.Sequential(
            nn.Linear(input_dim , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        # 将每个矩阵展成向量
        x = x[1]
        
        # 输入FFN进行计算
        x = self.ffn(x)

        # 将结果重塑回(batchsize, 10)的形式
        x = x.view(-1, 10)

        return x

class transformer_model(nn.Module):
    def __init__(self, encoderBlock, resnetBlock, outputBlock, input_dim) -> None:
        super().__init__()

        self.encoder_layer = nn.Sequential(*[encoderBlock(input_dim[i], input_dim[i + 1]) for i in range(len(input_dim) - 1)])

        self.output_layer = outputBlock(input_dim[-1])
    
    def forward(self, x):
        x = self.encoder_layer(x)

        x = self.resnet_layer(x)

        x = self.output_layer(x)
        return x

if __name__=="__main__": 
    # 初始化模型
    model = transformer_model(encoderBlock=encoder_block,outputBlock=output_block2,input_dim=[2000,1500,750,250,100])
    
    # 假设输入是形状为 [batch_size, sequence_length=10, embedding_dim=2000] 的one-hot向量
    input_one_hot = torch.randn(32, 10, 2000)  # batch_size=32
    
    # 对句子进行编码
    output = model(input_one_hot)
    
    # 输出形状应为 [batch_size, sequence_length=10, final_dim=10]
    print(output.shape)  # 应输出: torch.Size([32, 10, 10])