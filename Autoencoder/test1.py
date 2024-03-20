import torch
import torch.nn as nn
import torch.optim as optim
import AE_model

# 创建模型实例
model = AE_model.VAE()
loss_function = AE_model.loss_function

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 加载数据
file_path = 'data\GTR_val_tensor\one_hot_1_to_2000_batch0.pt'
data_tensor = torch.load(file_path)


# 假设我们有数据加载器 dataloader
for epoch in range(100):
    for batch_idx, data in enumerate(100):
        # 获取当前批次的数据
        inputs = data  # 假设data已经是one-hot编码后的形式
        
        # 前向传播并计算损失
        reconstructions, mu, logvar = model(inputs)
        loss = loss_function(reconstructions, inputs, mu, logvar, 2000)

        # 反向传播更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx % 100 == 0):
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(
                epoch+1, 100, batch_idx+1, len(100), loss.item()))