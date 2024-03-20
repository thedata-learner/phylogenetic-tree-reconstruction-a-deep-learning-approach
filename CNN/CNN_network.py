import torch
import torch.nn as nn
import torch.nn.functional as F
##################################################################################################################
'''
模型参数解释
encoder_type:   1.numeric   2.one_hot
task:   1.regression    2.classification

'''
#################################################################################################################
#子模块
#################################################encoder_layer##################################################
class encoder_layer1(nn.Module):
    #encoder_layer输出维度统一为(32,10,250)
    def __init__(self, encoder_type, ):
        super(encoder_layer1, self).__init__()
        #self.TS = nn.TransformerEncoderLayer(d_model=2000, nhead=10, dim_feedforward=2048, dropout=0.2, batch_first=True)

        if encoder_type=="one_hot":
            self.layer=nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(2,8),stride=(1,2),padding=(1,3)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(8,8),stride=(1,2),padding=(3,3)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=(1,2))
            )
        
        elif encoder_type=="numeric":#输入尺寸为(1,10,500)
                self.layer=nn.Sequential(
                    nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(5,3),stride=(1,1),padding=(2,1)),
                    nn.BatchNorm2d(18),
                    nn.ReLU(),
                    nn.Dropout(0.3),

                    nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(5,3),stride=(1,1),padding=(2,1)),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=(1,2))
                )
            
    def forward(self, x):
        #x = x.squeeze()
        #x = self.TS(x)
        #x = x.unsqueeze(dim=1)
        return self.layer(x)
###################################################ResNetLayer#################################################
class resnet_layer(nn.Module):
    # 输入维度是(1,32,10,250)
    # 输出维度是(1,32,10,10)
    def __init__(self, additional_layer):
        super(resnet_layer, self).__init__()
        # Adjust the resnet blocks to achieve the desired output shape

        self.additional_layer=self._make_additional_layer(additional_layer)

        self.resnet_block1 = self._make_resnet_block(32, 64, stride=(2, 5))  # Reducing height and width
        self.resnet_block2 = self._make_resnet_block(64, 64, stride=(2, 5))  # Further reducing
        self.resnet_block3 = self._make_resnet_block(64, 32, stride=(1, 1))  # Adjust to reach the final size


    def _make_additional_layer(self, additional_layer):
        return  nn.Sequential(*[identical_block() for _ in range(3)])
    
    def _make_resnet_block(self, in_channels, out_channels, stride):
        return BasicBlock(in_channels, out_channels, stride)

    def forward(self, x):
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != (1, 1) or in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class identical_block(nn.Module):
    def __init__(self, channels=32, kernel_size=3, stride=1, padding=1):
        super(identical_block, self).__init__()
        # 定义两个卷积层，其中输出通道数与输入通道数相同
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # 保存输入值（恒等连接）
        identity = x

        # 通过第一个卷积层后接批量归一化和ReLU激活函数
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # 通过第二个卷积层后接批量归一化
        out = self.conv2(out)
        out = self.bn2(out)

        # 将恒等连接的结果加到最后的输出上，再应用ReLU激活函数
        out += identity
        out = F.relu(out)

        return out
    
##################################################output_layer####################################################
class output_layer(nn.Module):
    # 输入维度是(batchsize,32,3,10)
    def __init__(self, task):
        super(output_layer, self).__init__()
        input_features = 32 * 3 * 10  # 展平后的输入
        self.task = task

        if task=="regression":#输出维度是torch.Size([batchsize,10])
            self.layer=nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_features,500),
                nn.ReLU(),
                nn.Linear(500,50),
                nn.ReLU(),
                nn.Linear(50,10)
            )

        elif task=="classification":#输出维度是torch.Size([batchsize,])
            self.layer=nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_features,100),
                nn.ReLU(),
                nn.Linear(100,),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        return self.layer(x)
    
###################################################################################################################

########################################################主模型######################################################
class network(nn.Module):
    def __init__(self, block_list, task, encoder_type ,additional_layer):
        super(network, self).__init__()
        
        self.encoder_layer =  self._make_encoder_layer(block_list[0],encoder_type,task)

        self.resnet_layer = self._make_resnet_layer(block_list[1], additional_layer)

        self.output_layer =  self._make_output_layer(block_list[2],task)

    def _make_encoder_layer(self,block,encoder_type, task):  
        return block(encoder_type=encoder_type)
    
    def _make_resnet_layer(self,block, additional_layer):
        return block(additional_layer)
    
    def _make_output_layer(self,block,task):
        return block(task)
    
    def forward(self, x):
        x = self.encoder_layer(x)
        x = self.resnet_layer(x)
        x = self.output_layer(x)
        return x

def resnet_for_matrix(task, encoder_type, additional_layer = 0):
    return network([encoder_layer1,resnet_layer,output_layer], task, encoder_type ,additional_layer)

##################################################################################################################
                                                #test#
##################################################################################################################
if __name__=="__main__":
    model=resnet_for_matrix(task="regression",encoder_type="one_hot", additional_layer=2)    
    input_data = torch.randn(10, 1, 10, 2000)
    print(model(input_data).shape)
