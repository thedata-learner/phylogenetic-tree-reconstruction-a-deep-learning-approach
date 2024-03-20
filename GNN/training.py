import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from transformer_model import transformer_model
from transformer_model import encoder_block
from transformer_model import output_block2
from tqdm import tqdm

class trainer():
    def __init__(self, device, model, epoch, criterion, optimizer, ):
        self.training_loss = []
        self.val_loss = []
        self.test_loss = []
        self.device = device
        self.model = model
        self.epoch = epoch
        self.criterion = criterion
        self.optimizer = optimizer


    def train(self, train_loader, val_loader, test_loader):
        for epoch in range(self.epoch):
            
            '''训练模式'''
            self.model.train()  
            running_loss_train = 0.0
            # 包装训练数据加载器，以显示进度条
            train_loader_with_progress = tqdm(enumerate(train_loader, 0), total=len(train_loader))
            for i, data in train_loader_with_progress:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs.squeeze())
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss_train += loss.item()

                # 更新进度条的描述
                train_loader_with_progress.set_description(f"Epoch {epoch+1} [Train] Loss: {running_loss_train/(i+1):.4f}")

            # 计算并保存训练集上的平均损失
            train_loss = running_loss_train / len(train_loader)
            self.training_loss.append(train_loss)

            '''验证模式'''
            self.model.eval()  # 设置模型为评估模式
            running_loss_val = 0.0
            # 包装验证数据加载器，以显示进度条
            val_loader_with_progress = tqdm(enumerate(val_loader, 0), total=len(val_loader))
            with torch.no_grad():  # 在评估模式下，不计算梯度
                for i, data in val_loader_with_progress:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.model(inputs.squeeze())
                    loss = self.criterion(outputs, labels)
                    running_loss_val += loss.item()

                    # 更新进度条的描述
                    val_loader_with_progress.set_description(f"Epoch {epoch+1} [val] Loss: {running_loss_val/(i+1):.4f}")

            # 计算并保存测试集上的平均损失
            val_loss = running_loss_val / len(val_loader)
            self.val_loss.append(val_loss)

            '''测试模式'''
            self.model.eval()  # 设置模型为评估模式
            running_loss_test = 0.0
            # 包装验证数据加载器，以显示进度条
            test_loader_with_progress = tqdm(enumerate(test_loader, 0), total=len(test_loader))
            with torch.no_grad():  # 在评估模式下，不计算梯度
                for i, data in test_loader_with_progress:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.model(inputs.squeeze())
                    loss = self.criterion(outputs, labels)
                    running_loss_test += loss.item()

                    # 更新进度条的描述
                    test_loader_with_progress.set_description(f"Epoch {epoch+1} [Test] Loss: {running_loss_test/(i+1):.4f}")

            # 计算并保存测试集上的平均损失
            test_loss = running_loss_test / len(test_loader)
            self.test_loss.append(test_loss)

        
            if epoch == 0 or val_loss < best_val_loss:# 选出当前最好的模型
                best_val_loss = val_loss
                best_model_state_dict = self.model.state_dict()
    
        print(f'Epoch {epoch+1} Train loss: {train_loss:.3f},Val loss: {val_loss:.3f},Test loss: {test_loss:.3f}')
        self.model.load_state_dict(best_model_state_dict)

def split_data(file_path, ratio=0.8):
    # 加载数据部分训练数据
    training_tensor = torch.load(file_path)
    
    # 计算要提取的数据数量
    data_size = len(training_tensor)
    split_size = int(ratio * data_size)
    
    # 使用 random_split 函数划分数据
    train_data, _ = random_split(training_tensor, [split_size, data_size - split_size])
    
    return train_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
file_path = 'data\GTR_training_tensor\one_hot_regression_1_to_10000_batch0.pt'
training_tensor = split_data(file_path=file_path, ratio=1.0)
print("training_tensor")
file_path = 'data/GTR_val_tensor/one_hot_1_to_2000_batch0.pt'
val_tensor = split_data(file_path=file_path ,ratio=0.2)
print("val_tensor")
file_path = 'data/GTR_testing_tensor/one_hot_regression_1_to_500_batch0.pt'
test_tensor = torch.load(file_path)
print("testing_tensor")

train_loader = DataLoader(training_tensor, batch_size=512, shuffle=True)
val_loader = DataLoader(val_tensor, batch_size=256, shuffle=True)
test_loader = DataLoader(test_tensor, batch_size=256, shuffle=True)


# 创建模型实例
model = transformer_model(encoderBlock=encoder_block,outputBlock=output_block2,input_dim=[2000,1000,1000,300]).to(device)

# 定义优化器
optimizer = Adam(model.parameters(), lr=0.01)
epochs = 100

# 创建数据加载器
train_loader = DataLoader(training_tensor, batch_size=512, shuffle=True)
val_loader = DataLoader(val_tensor, batch_size=256, shuffle=True)
test_loader = DataLoader(test_tensor, batch_size=256, shuffle=True)

criterion = nn.MSELoss()#回归


loss_fn = nn.MSELoss()
train = trainer(device=device, epoch=epochs, model=model, criterion=criterion, optimizer=optimizer)
train.train(train_loader=train_loader,val_loader=val_loader ,test_loader=test_loader)
train.log(path="model",name=f'TS_6_layer')
