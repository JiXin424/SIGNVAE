import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from text_encoder import WordEncoder, encode_word  # 确保你有这个文件
from motion_vae import MotionVAE  # 确保你有这个文件
from data_process import load_dataset
from FlagEmbedding import BGEM3FlagModel

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 对齐损失函数
def align_loss(text_emb, motion_emb):
    return nn.MSELoss()(text_emb, motion_emb)

# 训练对齐网络
def train_alignment_network(motion_vae, word_encoder, batch_size, num_epochs_align, bge_model):
    motions, labels = load_dataset("data")
    motions = torch.tensor(motions, dtype=torch.float32).view(motions.shape[0], -1)
    labels_encoded = torch.stack([encode_word(word, bge_model) for word in labels])

    dataset = TensorDataset(motions, labels_encoded)
    motion_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    align_optimizer = optim.Adam(word_encoder.parameters(), lr=1e-2)

    print("Training Alignment Network...")
    for epoch in range(num_epochs_align):
        total_align_loss = 0
        for batch_idx, (batch, text_vectors) in enumerate(motion_loader):
            batch = batch.to(torch.float32).to(device)
            text_vectors = text_vectors.to(torch.float32).squeeze(1).to(device)
            
            text_vectors = word_encoder(text_vectors)
            align_optimizer.zero_grad()

            with torch.no_grad():  # 确保在评估模式下使用 VAE
                recon, mu, logvar = motion_vae(batch)
                mu_expanded = mu.unsqueeze(1)  # 现在形状为 [2, 1, 8]
                logvar_expanded = logvar.unsqueeze(1)  # 现在形状为 [2, 1, 8]
                motion_vectors = torch.cat((mu_expanded, logvar_expanded), dim=1)
                
            loss_align = align_loss(text_vectors, motion_vectors)
            loss_align.backward()
            align_optimizer.step()
            total_align_loss += loss_align.item()

        print(f"Epoch {epoch+1}: Align Loss = {total_align_loss / len(motion_loader)}")

    # 保存对齐模型
    torch.save(word_encoder.state_dict(), "word_encoder.pth")

# 主程序入口
if __name__ == "__main__":
    bge_model = BGEM3FlagModel(rf'D:\全部资料\STORE\PersonalData\lzq\Study\Project\SLKG\code\SignKG\thirdparty\bge\bge-m3',
                                    use_fp16=True)
    # 初始化 VAE 和文本编码器
    motion_vae = MotionVAE().to(device)
    word_encoder = WordEncoder().to(device)

    # 加载预训练的 VAE 模型
    motion_vae.load_state_dict(torch.load("motion_vae.pth"))
    # 冻结 VAE 参数
    for param in motion_vae.parameters():
        param.requires_grad = False

    batch_size = 2
    num_epochs_align = 500

    # 开始训练对齐网络
    train_alignment_network(motion_vae, word_encoder, batch_size, num_epochs_align, bge_model)
