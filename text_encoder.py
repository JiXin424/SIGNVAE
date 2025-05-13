
import torch
import torch.nn as nn
from FlagEmbedding import BGEM3FlagModel


class WordEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_slots=2, embedding_dim=256):
        super(WordEncoder, self).__init__()

        self.num_slots = num_slots
        self.embedding_dim = embedding_dim
        output_dim = num_slots * embedding_dim  # 输出总维度
        
        # 定义三层全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # ReLU激活函数
        self.relu = nn.ReLU()

        
    def forward(self, x):
        # 通过三层全连接层传递嵌入向量
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        x = x.view(x.size(0), self.num_slots, self.embedding_dim)
        return x
    

def encode_word(word, bge_model):
    """输入文本 word，返回编码后的 1024 维向量"""
    init_embeddings = bge_model.encode([word])['dense_vecs']  # (1, 1024)
    init_embeddings = torch.tensor(init_embeddings, dtype=torch.float32)
    return init_embeddings


if __name__ == "__main__":
    word_encoder = WordEncoder()
    bge_model = BGEM3FlagModel(rf'D:\全部资料\STORE\PersonalData\lzq\Study\Project\SLKG\code\SignKG\thirdparty\bge\bge-m3',
                                use_fp16=True)
    # Test encoding a word
    word = "example_word"
    labels_encoded = encode_word(word, bge_model)
    encoded_vector = word_encoder(labels_encoded)
    print(f"Encoded vector for '{word}': {encoded_vector}")
