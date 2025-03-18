import os
import requests
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt


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


class EmbeddingLayer(nn.Module):
    """复杂将用户的输入向量化，并且完成位置信息的添加"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


class MultiHeadAttensionLayer(nn.Module):
    """将输入复制出QKV，然后切分为多个head，进行一系列注意力的计算。最后拼接到一起经过Wo后变为output"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


class SingleTransformerBlock(nn.Module):
    """经过前面的 MultiHeadAttension 后，残差连接+经过层归一化+FeedForwar+残差+层归一化"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


class FeedForwardLayer(nn.Module):
    """transformer block最后需要用到FeedForward"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


class MultiTransformerBlock(nn.Module):
    """循环调用前面的 SingleTransformerBlock num_blocks次"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


class LinearLayer(nn.Module):
    """经过所有Transformer Block后，还需要经过一个Linear层+softmax，完成向[context_length, d_model] => [context_length, vob_size]的转化"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass
