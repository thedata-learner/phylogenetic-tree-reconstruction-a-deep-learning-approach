import torch
from torch.utils.data import DataLoader,random_split
import CNN_network
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import json

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
                outputs = self.model(inputs)
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
                    outputs = self.model(inputs)
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
                    outputs = self.model(inputs)
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

    def draw(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_loss, label='Train Loss')
        plt.plot(self.val_loss, label='val Loss')
        plt.plot(self.test_loss, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train and Test Loss Over Epochs')
        plt.legend()
        plt.show()
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    


    def log(self, path, name):
        # Save the losses
        losses = {
            'training_loss': self.training_loss,
            'val_loss': self.val_loss,
            'test_loss': self.test_loss
        }
        with open(f'{path}/{name}_losses.json', 'w') as f:
            json.dump(losses, f)

        # Save the model
        model_path = f'{path}/{name}_model.pth'
        torch.save(self.model.state_dict(), model_path)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, targets):
        # 假设outputs和targets是形状相同的张量列表，每个元素代表一个样本的输出和目标标签
        if not isinstance(outputs, list) or not isinstance(targets, list):
            raise ValueError("Outputs and targets should be lists of tensors.")
        
        if len(outputs) != 10 or len(targets) != 10:
            raise ValueError("The length of both outputs and targets should be 10.")
        
        loss_list = [F.cross_entropy(output, target) for output, target in zip(outputs, targets)]
        
        # 计算平均损失
        average_loss = sum(loss_list) / len(loss_list)
        
        return average_loss


def split_data(file_path, ratio=0.8):
    # 加载数据部分训练数据
    training_tensor = torch.load(file_path)
    
    # 计算要提取的数据数量
    data_size = len(training_tensor)
    split_size = int(ratio * data_size)
    
    # 使用 random_split 函数划分数据
    train_data, _ = random_split(training_tensor, [split_size, data_size - split_size])
    
    return train_data

####################################################################################################################
def main():
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

    model=CNN_network.resnet_for_matrix(encoder_type="one_hot",task="regression",additional_layer=7).to(device)
        
    criterion = nn.MSELoss()#回归
    #criterion = CustomLoss()#分类

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 100

    train = trainer(device=device, epoch=epochs, model=model, criterion=criterion, optimizer=optimizer)
    train.train(train_loader=train_loader,val_loader=val_loader ,test_loader=test_loader)
    train.log(path="model",name=f'resnt_6_layer')


if __name__ == "__main__":
    main()