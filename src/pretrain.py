import math

import functorch.dim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from loguru import logger
from dataclasses import dataclass

torch.manual_seed(1024)

# define GPT param
@dataclass
class GPTConfig:
    block_size: int=512 # max_seq
    batch_size: int=12
    n_layer: int=12
    n_head: int=12
    n_embd: int=768 # hidden_dim,hidden_size,same with emb_size for tie_embedding_weight
    hidden_dim: int=n_embd
    dropout: float=0.1
    head_size:int=n_embd//n_head
    vocab_size:int=50274 # same with GPT2

# define GPT Struct
# 1.define single head attention
class SingleHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.key=nn.Linear(config.hidden_dim,config.head_size)
        self.value=nn.Linear(config.hidden_dim,config.head_size)
        self.query=nn.Linear(config.hidden_dim,config.head_size)

        # attention mask by register_buffer
        self.register_buffer(
            "attention_mask",
            torch.tril(
                torch.ones(config.block_size,config.block_size)
            )
        )
        self.dropout=nn.Dropout(config.dropout)

    # forward layer
    def forward(self,x):
        batch_size,seq_len,hidden_dim=x.size()
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        weight=q @ k.transpose(-2,-1)
        weight=weight.masked_fill(
            self.attention_mask[:seq_len,:seq_len]==0,
            float('-inf')
        )
        weight=F.softmax(weight,dim=-1)/math.sqrt(self.head_size)
        weight=self.dropout(weight)
        out=weight @ v
        return out

# 2.define multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.heads=nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_head)
            ]
        )
        self.proj =nn.Linear(config.hidden_dim,config.hidden_dim)
        self.dropout=nn.Dropout(config.dropout)

    def forward(self,x):
        output=torch.cat(
            [
                h(x) for h in self.heads
            ],
            dim =-1
        )
        output=self.proj(output)
        output=self.dropout(output)
        return output

# 3. define feed forward
class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(config.hidden_dim,4*config.hidden_dim),
            nn.GELU(),
            nn.Linear(4*config.hidden_dim,config.hidden_dim),
            nn.Dropout(config.dropout)
        )
    def forward(self,x):
        return self.net(x)

# 4.block
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.att=MultiHeadAttention(config) # mha
        self.ffn=FeedForward(config)
        self.ln1=nn.LayerNorm(config.hidden_dim)
        self.ln2=nn.LayerNorm(config.hidden_dim)

    def forward(self,x):
        x=x+self.att(self.ln1(x))
        x=x+self.ffn(self.ln2(x))
        return x

# 5.GPT embedding position norm mlp block
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.token_embedding_table=nn.Embedding(config.vocab_size,config.n_embd)
        self.position_embedding_table=nn.Embedding(config.block_size,config.n_embd)
        self.blocks=nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final=nn.LayerNorm(config.n_embd)
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)

        self.token_embedding_table.weight=self.lm_head.weight

    def _init_weight(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self,idx,targets=None):
        batch,seq_len=idx.size()
        token_emb=self.token_embedding_table(idx)
        pos_emb=self.position_embedding_table(
            torch.arange(seq_len,device=idx.device)
        )
        x=token_emb+pos_emb
        x=self.blocks(x)
        x=self.ln_final(x)
        logits=self.lm_head(x)
        if targets is None:
            loss=None
        else:
            batch,seq_len,vocab_size=logits.size()
            logits=logits.view(batch*seq_len,vocab_size)
            targets=targets.view(batch*seq_len)
            loss=F.cross_entropy(logits,targets)
        return logits,loss

    def generate(self,idx,max_new_tokens):
        pass


# 写一个 dataset，为了 Dataloader 准备
class MyDataset(Dataset):
    def __init__(self, path, block_size=512):
        # 我的数据在 /root/fs/mobvoi_seq_monkey_general_open_corpus.jsonl 中，
        # 读取前 1000 行
        import tiktoken # GPTtokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size

        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

        import json

        self.encoded_data = []

        self.max_lines = 1000
        raw_data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])

        # 将长文本分割成训练样本
        for i in range(0, len(full_encoded), self.block_size):
            # 多取一个 Token 作为目标
            chunk = full_encoded[i:i + self.block_size + 1]
            # 如果长度不够，用 eos_token 填充
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        """将文本编码为token IDs"""
        return self.enc.encode(text)

    def decode(self, ids):
        """将token IDs解码为文本"""
        return self.enc.decode(ids)