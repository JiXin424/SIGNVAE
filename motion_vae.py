import torch
import torch.nn as nn


class MotionVAE(nn.Module):
    def __init__(self, input_dim=150 * 137 * 2, hidden_dim=1024, latent_dim=512):
        super(MotionVAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        # 均值和方差的维度设置，默认2
        self.mean_layer = nn.Linear(latent_dim, 256)
        self.logvar_layer = nn.Linear(latent_dim, 256)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
        
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    # 未使用std进行重参数化
    # def reparameterization(self, mean, logvar):
    #     epsilon = torch.randn_like(logvar).to(device)      
    #     z = mean + logvar*epsilon
    #     return z
    
    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)  # 计算标准差
        epsilon = torch.randn_like(std).to(mean.device)      
        z = mean + std * epsilon
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
    
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = MotionVAE().to(device)
    test_input = torch.randn(2, 150 * 137 * 2,  device=device)  # 创建随机输入
    recon, mu, logvar = vae(test_input)

    if recon is None:
        print("Error: recon is None!")
    else:
        print("VAE forward pass successful.")
