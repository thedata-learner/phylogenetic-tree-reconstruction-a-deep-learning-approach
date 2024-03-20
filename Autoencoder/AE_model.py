import torch
import torch.nn as nn
import torch.optim as optim



class VAE(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dim):
        super(VAE, self).__init__()

        self.input_size = input_size # 输入的序列长度
        self.latent_dim = latent_dim # 潜在空间维度
        self.hidden_dim = hidden_dim # 隐藏层神经元数量

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  
        )

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_size * 4), 
            nn.Sigmoid() 
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, self.input_size)  # 将数据展平以适应全连接层
        mu, logvar = self.encoder(x).chunk(2, dim=-1)  # 分解出均值和对数方差
        z = self.reparameterize(mu, logvar)  # 通过重新参数化技巧采样
        reconstruction = self.decoder(z)  # 通过解码器进行重构
        return reconstruction, mu, logvar
    
def loss_function(recon_x, x, mu, logvar, input_size):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')  # 二元交叉熵损失函数计算重构误差
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # 计算KL散度
    KLD *= input_size  # 对每个样本的每个位置进行归一化
    return BCE + KLD

if __name__ == "__main__":
    x = torch.rand(1, 2000)

    model = VAE()

    print(model(x))