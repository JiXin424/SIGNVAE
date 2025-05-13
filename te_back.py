from FlagEmbedding import BGEM3FlagModel
import torch
import torch.nn as nn
import torchvision.models as models


class WordEncoderResNet(nn.Module):
    def __init__(self, input_dim=1024, output_dim=512):
        super(WordEncoderResNet, self).__init__()
        self.resnet = models.resnet18(weights=None)  # 加载 ResNet18
        
        # 修改 ResNet18 第一层，使其适应 1D 输入
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)
        
        # 修改 ResNet18 最后的全连接层，使其输出 512 维向量
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(3)  # 变成 (batch, 1, 32, 32) 适应 ResNet
        x = self.resnet(x)
        return x

word_encoder = WordEncoderResNet()

def word_init(word):
    model = BGEM3FlagModel(rf'D:\全部资料\STORE\PersonalData\lzq\Study\Project\SLKG\code\SignKG\thirdparty\bge\bge-m3',
                           use_fp16=True)
    init_embeddings = model.encode([word])['dense_vecs']
    return torch.tensor(init_embeddings, dtype=torch.float32)

def encode_word(word):
    init_embeddings = word_init(word)  # (1, 1024)
    encoded_embedding = word_encoder(init_embeddings)  # (1, 512)
    return encoded_embedding