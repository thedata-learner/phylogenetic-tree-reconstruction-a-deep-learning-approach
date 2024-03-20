import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
###########################################################################################
class node_aggregation(torch.nn.Module):
    '''
    此部分为图节点聚合任务
    '''
    def __init__(self, in_dim, out_dim ) -> None:
        super(node_aggregation, self).__init__()
        self.conv1 = GCNConv(in_channels=in_dim, out_channels=1000)
        self.conv2 = GCNConv(in_channels=1000, out_channels=out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class edge_processer(torch.nn.Module):
    '''
    此部分为边节点回归或分类任务
    '''
    def __init__(self, in_dim, task) -> None:
        super(edge_processer, self).__init__()
        if task == "regression":
            self.function = torch.nn.Linear(2 * in_dim, 1)
        
        elif task == "classification":
            self.function = F.softmax(torch.nn.Linear(2 * in_dim, 8), dim=1)

    def forward(self, x, edge_index):
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_weights = self.function(edge_features)

        return edge_weights

#############################################################################################
class GNN(torch.nn.Module):
    def __init__(self, block_list, in_dim, out_dim, task="regression") -> None:
        '''

        '''
        super(GNN, self).__init__()
        #节点特征聚合
        self.node_aggregation_layer = self.make_node_aggregation_layer(block_list[0], in_dim, out_dim)

        #边回归或者分类器
        self.edge_processer_layer = self.make_edge_processer_layer(block_list[1], out_dim, task)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.node_aggregation_layer(x, edge_index)
        edge_features = self.edge_processer_layer(x, edge_index)

        return edge_features
    
    def make_node_aggregation_layer(self, block, in_dim, out_dim):
        return block(in_dim, out_dim)
    
    def make_edge_processer_layer(self, block, out_dim, task):
        return block(out_dim, task)
    
###########################################################################################
def creat_GNN():
    return GNN(block_list=[node_aggregation, edge_processer], in_dim=2000, out_dim=250)

if __name__=="__main__":
    #初始化数据
    num_nodes = 10
    num_node_features = 2000
    x = torch.randn((num_nodes, num_node_features))

    # 创建全连接图的边索引
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
    data = Data(x=x, edge_index=edge_index)

    #初始化模型
    model = creat_GNN()
    #向前传播
    edge_weights = model(data)
