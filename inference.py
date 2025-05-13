import torch
from motion_vae import MotionVAE
from text_encoder import WordEncoder, encode_word
from FlagEmbedding import BGEM3FlagModel
from data_process import denormalize_motion_sequence, visualize_motion_sequence
import os
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
motion_vae = MotionVAE().to(device)
word_encoder = WordEncoder().to(device)

motion_vae.load_state_dict(torch.load("motion_vae.pth"))
word_encoder.load_state_dict(torch.load("word_encoder.pth"))

motion_vae.eval()
word_encoder.eval()

def generate_motion_from_text(word,bge_model):
    text_init = encode_word(word,bge_model).to(device) # 生成文本嵌入

    # text_vector = torch.tensor(text_init, dtype=torch.float32).unsqueeze(0).to(device)  # 添加批次维度
    text_vector = word_encoder(text_init)

    motion_vector = text_vector  # 直接使用文本嵌入作为 VAE 输入
    mean = motion_vector[:, 0, :]
    logvar = motion_vector[:, 1, :]
    z = motion_vae.reparameterization(mean, logvar).to(device)
    generated_motion = motion_vae.decoder(z)  # 解码手语动作
    return generated_motion.view(150, 137, 2).detach().cpu().numpy()

# 示例
if __name__ == "__main__":
    bge_model = BGEM3FlagModel(rf'D:\全部资料\PersonalData\lzq\Study\Project\SLKG\code\SignKG\thirdparty\bge\bge-m3',
                                    use_fp16=True)
    motion_sequence = generate_motion_from_text("101斑点狗",bge_model)
    motion_sequence_re = denormalize_motion_sequence(motion_sequence)
    visualize_motion_sequence(motion_sequence_re)
