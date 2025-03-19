import math
import os
import requests
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken
from torch.nn import functional as F

d_model = 64  # dimention
batch_size = 4
context_length = 16
num_blocks = 8
num_heads = 4
dropout = 0.1


def set_cuda():
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    else:
        torch.set_default_device("cpu")

    x = torch.randn(3, 3)
    print(x.device)  # 如果有 GPU，则输出 `cuda:0`


def load_data():
    if not os.path.exists("sales_textbook.txt"):
        url = "https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt"
        with open("sales_textbook.txt", "w") as f:
            f.write(requests.get(url).text)

    with open("sales_textbook.txt", "r", encoding="utf-8") as f:
        text = f.read()

    return text


set_cuda()
text = load_data()
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text=text)
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)
max_token_id = tokenized_text.max().item()

# 根据8 2的比例切分训练数据集和验证数据集
split_idx = int(len(tokenized_text) * 0.8)
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]


class FeedForward(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.ffn = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=d_model * 4, out_features=d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class Attension(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 因为这里是单个头，所以这里的Wq也是根据head的个数初始化部分就行
        self.Wq = nn.Linear(d_model, d_model // num_heads)
        self.Wk = nn.Linear(d_model, d_model // num_heads)
        self.Wv = nn.Linear(d_model, d_model // num_heads)

    def forward(self, x):
        # [batch_size, context_length, d_model] @ [batch_size, d_model, d_model // num_heads]
        # ==> [batch_size, context_length, d_model // num_heads]
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # [batch_size, context_length, d_model // num_heads] @ [batch_size, d_model // num_heads, context_length]
        # ==> [batch_size, context_length, context_length]
        attention_score = Q @ K.transpose(-2, -1)

        # scale
        attention_score /= math.sqrt(d_model // num_heads)

        # mask
        mask = torch.triu(
            torch.ones_like(attention_score, dtype=torch.bool), diagonal=1
        )
        attention_score = attention_score.masked_fill(mask, -float("inf"))

        # softmax, [batch_size, context_length, context_length]
        attention_score = F.softmax(attention_score)

        # [batch_size, context_length, context_length] @ batch_size, context_length, d_model // num_heads]
        # ==> [batch_size, context_length, d_model // num_heads]
        A = attention_score @ V
        return A


class MultiAttension(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Attension() for _ in range(num_heads)])
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 每个head计算出来的维度都是：[batch_size, context_length, d_model // num_heads]
        # 所以需要在最后一个维度进行拼接，也就是dim = -1
        # 拼接后的维度：[batch_size, context_length, d_model]
        A = torch.cat([h(x) for h in self.heads], dim=-1)

        # 最后还需要经过一个[batch_size, d_model, d_model]的权重矩阵
        # [batch_size, context_length, d_model] @ [batch_size, d_model, d_model]
        # ==> [batch_size, context_length, d_model]
        # ==> 维持和我们输入一样的维度了！
        return A @ self.Wo


class TransformerBlock(nn.Module):
    """这里是单个Transformer Block的逻辑，这样的Block一共会有 num_blocks 个"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_head_attention = MultiAttension()
        self.feed_forward = FeedForward()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 假设这里的x已经是加上位置信息后的[batch_size, context_length, d_model]
        # 1. 先经过一个MultiHeadAttension
        output = self.multi_head_attention(x)
        # 2. 残差连接
        output += x
        # 3. 经过层归一化
        output = self.layer_norm_1(output)
        # 4. feed forward
        after_feed_forward = self.feed_forward(output)
        # 5. 残差连接, 加上feed forward之前的输入
        output = after_feed_forward + output
        # 6. 再经过一次层归一化
        return self.layer_norm_2(output)


class TransformerLanguageModel(nn.Module):
    """最终的多层Transformer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock() for _ in range(num_blocks)]
        )
        # 最后将[batch_size, context_length, d_model] 转为 [batch_size, context_length, vocab_size]的矩阵
        self.final_linear = nn.Linear(d_model, max_token_id + 1)

    def forward(self, idx, targets=None):
        """注意这里的入参不一样了

        Args:
            idx: token代表的id数组输入
            targets: 预期输出，如果传了就是训练模式，可以根据这个target计算出loss
        """
        # 1. embedding

        # 2. calc positional matrix

        # TODO