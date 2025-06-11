import torch
from sympy.core.random import random
from torch import nn
import torch.nn.functional as F
import math

from torch.nn import Embedding

# 输出词汇表索引转换为指定纬度
class TokenEmbedding(nn.Embedding):
    def __init__(self,vocb_size,d_model):
        super(TokenEmbedding,self).__init__(vocb_size,d_model,padding_idx=1)

class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,max_len,device):
        # d_model，一个 token 或位置的嵌入是一个 d_model 维的向量
        # 每个词的语义特征空间大小；
        # 每个位置编码向量的维度；
        # Transformer 中各层隐状态的维度。
        super(PositionalEmbedding,self).__init__()
        # 1 初始化全0矩阵
        self.encoding=torch.zeros(max_len,d_model,device=device)
        self.encoding.requires_grad=False

        # 2 定义序列
        pos=torch.arange(0,max_len,device=device)# 构造一个向量，表示每个token在序列中的位置
        # 转换为float ，并转换为张量
        pos=pos.float().unsqueeze(dim=1)# 转换为float类型并升纬
        _zi=torch.arange(0,d_model,step=2,device=device).float()

        # 3 计算位置编码
        self.encoding[:,0::2]=torch.sin(pos/10000**(_zi/d_model))
        self.encoding[:, 1::2] = torch.cos(pos / 10000 ** (_zi / d_model))

    # 4 前向传播过程forward
    def forward(self,x):
        # 获取x大小
        batch_size,seq_len=x.size()
        # batch_size: 表示有多少个句子或样本。
        # seq_len: 每个句子中有多少个 token。
        # 返回编码矩阵中的seq_len长度的位置编码序列
        # 根据输入的序列长度，返回对应长度的“位置嵌入矩阵”。
        return self.encoding[:seq_len,:]
        """
        self.encoding 是一个事先准备好的 [max_len, d_model] 的位置编码矩阵（在 __init__() 中生成）
        self.encoding[:seq_len, :] 表示从第 0 行开始，取前 seq_len 行的编码（也就是前 seq_len 个位置的编码向量）
        返回的是一个形状为 [seq_len, d_model] 的张量
        """

class TransformerEmbedding(nn.modules):
    def __init__(self,vocb_size,d_model,max_len,drop_prob,device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb=TokenEmbedding(vocb_size,d_model)
        self.pos_emb=PositionalEmbedding(d_model,max_len, device)
        self.drop_out=nn.Dropout(p=drop_prob)# 训练过程中随机丢弃一些神经元，避免过拟合

    def forward(self,x):
        tok_emb=self.tok_emb(x)
        pos_emb=self.pos_emb(x)
        return self.drop_out(tok_emb+pos_emb)

# https://www.bilibili.com/video/BV1nXjEzmEWC?spm_id_from=333.788.player.switch&vd_source=c3c4336342cea4cb20fdd7ac31bd0079&p=2
if __name__ == '__main__':
    random_torch=torch.rand(4,4)
    print(random_torch)