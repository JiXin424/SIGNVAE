import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from data_process import load_dataset
from motion_vae import MotionVAE
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练 VAE
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # print("输入数据范围:", x.min().item(), x.max().item())
    # print("重构数据范围:", recon_x.min().item(), recon_x.max().item())

    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # kl_loss = torch.clamp(kl_loss, min=-1e6, max=1e6)  # 限制 KL 损失范围

    # if torch.isnan(recon_loss).any() or torch.isinf(recon_loss).any():
    #     print(f"recon_loss has NaN or Inf! Value: {recon_loss.item()}")
    
    # if torch.isnan(kl_loss).any() or torch.isinf(kl_loss).any():
    #     print(f"kl_loss has NaN or Inf! Value: {kl_loss.item()}")

    return recon_loss + beta * kl_loss


# 加载数据
motions, labels = load_dataset("data")
# max_value = motions.max()
# min_value = motions.min()
# print("最大值:", max_value.item())
# print("最小值:", min_value.item())

motions = torch.tensor(motions, dtype=torch.float32).view(motions.shape[0], -1) # 变为 (num_samples, 150*137*2)

# 创建 DataLoader
dataset = TensorDataset(motions)
batch_size = 2
motion_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化 VAE
motion_vae = MotionVAE().to(device)
vae_optimizer = optim.Adam(motion_vae.parameters(), lr=4e-5)
num_epochs_vae = 500

#  训练 VAE 网络
print("Training VAE...")
for epoch in range(num_epochs_vae):
    total_vae_loss = 0
    for batch_idx, batch in enumerate(motion_loader):
        batch = batch[0].view(batch_size, 150 * 137 * 2).to(torch.float32).to(device)

        vae_optimizer.zero_grad()

        recon, mu, logvar = motion_vae(batch)
        recon = recon.to(torch.float32).to(device)
        loss_vae = vae_loss(recon, batch, mu, logvar)
        loss_vae.backward()
        # torch.nn.utils.clip_grad_norm_(motion_vae.parameters(), max_norm=1.0)  # 梯度裁剪
        vae_optimizer.step()
        total_vae_loss += loss_vae.item()

    average_loss = total_vae_loss / len(motion_loader)

    formatted_loss = "{:.2e}".format(average_loss)
    print(f"Epoch {epoch+1}: VAE Loss = {formatted_loss}")

# 保存 VAE 模型
torch.save(motion_vae.state_dict(), "motion_vae.pth")
