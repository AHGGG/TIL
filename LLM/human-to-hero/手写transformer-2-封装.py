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
learning_rate = 0.001  # 学习率
max_iters = 2000  # 训练的总次数
temperature = 1.0


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
    def __init__(self):
        super().__init__()
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
    def __init__(self):
        super().__init__()
        # 因为这里是单个头，所以这里的Wq也是根据head的个数初始化部分就行
        self.Wq = nn.Linear(d_model, d_model // num_heads)
        self.Wk = nn.Linear(d_model, d_model // num_heads)
        self.Wv = nn.Linear(d_model, d_model // num_heads)

    def forward(self, x):
        # [batch_size, context_length, d_model] @ [batch_size, d_model, d_model // num_heads]
        # ==> [batch_size, context_length, d_model // num_heads]
        # self.Wq 是 nn.Linear 层，不是一个 Tensor 或 nn.Parameter, 所以需要像函数一样调用
        Q = self.Wq(x)  # x @ self.Wq
        K = self.Wk(x)  # x @ self.Wk
        V = self.Wv(x)  # x @ self.Wv

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
        attention_score = F.softmax(attention_score, dim=-1)

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
        return self.Wo(A)  # A @ self.Wo


class TransformerBlock(nn.Module):
    """这里是单个Transformer Block的逻辑，这样的Block一共会有 num_blocks 个"""

    def __init__(self):
        super().__init__()
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

    def __init__(self):
        super().__init__()
        # 第一个参数为embedding字典的大小，这里取+1是确保到时候max_token_id这个值能够从下标为0开始的数组中拿到值
        # 第二个参数为embedding的维度，也就是我们一个字多少个维度，这里用d_model
        self.token_embedding_lookup_table = nn.Embedding(max_token_id + 1, d_model)
        # 注意这里的是nn.Sequential, 而不是nn.ModuleList
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(num_blocks)]
        )
        # 最后将[batch_size, context_length, d_model] 转为 [batch_size, context_length, vocab_size]的矩阵
        self.final_linear = nn.Linear(d_model, max_token_id + 1)

    def forward(self, idx, targets=None):
        """注意这里的入参不一样了

        Args:
            idx: token代表的id数组输入（有batch_size）, 这里已经已经tokenizer处理了
            targets: 预期输出，如果传了就是训练模式，可以根据这个target计算出loss
        """
        # 注意，这里idx是[batch_size, idx_length]的维度！！！
        B, idx_length = idx.shape

        # from [batch_size, idx_length] to [batch_size, idx_length, d_model]
        embedding_input = self.token_embedding_lookup_table(idx)

        # 1. calc positional matrix
        # 1.1 先确定我们position encoding这个矩阵的大小，因为要和我们输入的维度一样，所以是：[idx_length, d_model]
        # 1.1 进行初始化，因为位置编码信息是一样的常数，所以这里只需要计算一个batch的出来，其他的batch都直接相加就行
        position_encoding_lookup_table = torch.zeros(idx_length, d_model)

        # 1.2 计算位置编码matrix
        # 从每一行的维度看，公式中的pos每次+1的递增。
        # 从每一列的维度遍历，i的取值范围从[0, d_model / 2]也就是[0, 32], 一个i对应两个元素，第一个用sin，第二个用cos
        for pos in range(idx_length):
            for i in range(0, d_model, 2):
                sin_index = i
                cos_index = i + 1
                position_encoding_lookup_table[pos][sin_index] = math.sin(
                    pos / (math.pow(10000, 2 * i / d_model))
                )
                position_encoding_lookup_table[pos][cos_index] = math.cos(
                    pos / (math.pow(10000, 2 * i / d_model))
                )

        # 1.2 optional
        # 使用torch的api可以简化前面的计算，我感觉还是前面的好理解，不用受torch api的影响
        # position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        # result = torch.sin(position * div_term)
        # position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        # position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)

        # 2. 将embeddin后的输入 + position encode后的结果 ==> 输入中就包含了位置信息
        # 2.1 [batch_size, idx_length, d_model] + [idx_length, d_model]在torch中是可以相加的，结果的维度还是[batch_size, idx_length, d_model]
        x = embedding_input + position_encoding_lookup_table

        # 3. 经过transformer_blocks
        x = self.transformer_blocks(x)

        # 4. 经过最后的线性层, logits ==> [batch_size, idx_length, vocab_size]
        logits = self.final_linear(x)

        # 判断如果targets不为空，那么我们需要计算出loss
        if targets is not None:
            # B: batch_size, T: timestamp, current context_length(idx_length) ==> 序列长度, C: Channels(dimensions） ==> 类别数量
            B, T, C = logits.shape
            # 交叉熵损失函数`F.cross_entropy(input=logits, target=targets_reshaped)`期望输入的形状是 (N, C) 和 (N)，其中 N 是样本数量，C 是类别数量。
            # 我的理解: 为什么target里不需要考虑C(类别数), 因为target代表的意思就是预期的类别, 而我们的输入里放了C个类别的概率. 所以输入的每个序列里的某一个元素, 都有一个预期的类别, 这样target其实就相当于降维了.
            # 将 logits 和 targets 重塑为 (B * T, C) 和 (B * T)
            logits_reshaped = logits.view(B * T, C)  # 重塑为 (B * T, C)
            # 其实targets可以理解成每个token都有一个预期的答案，但是这个在logits中是d_model个概率表示的（最大的就是计算出来的），所以targets里自然就缺少一个维度（dimension）
            targets_reshaped = targets.view(B * T)   # 重塑为 (B * T)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_token: int):
        """forward用于训练, 这个generate方法用于生成
        
        Args:
            idx: 这里是输入的文本, 转成token后的数组, 文本可能非常长
            max_new_token: 要补全多少个token
        """
        # idx ==> [batch_size, idx_length]
        for _ in range(max_new_token):
            # 第一个维度不管, 第二个维度按照上下文窗口取出对应长度的文本
            idx_ = idx[:, -context_length:]
            # logits: [batch_size, context_length, vocab_size]
            logits, loss = self(idx_)
            # 第一个维度批次不管, 拿到第二个维度context_length中的最后一个token对应的向量 [batch_size, vocab_size]
            last_token_logits = logits[:, -1, :]
            # 拿到logits后再经过一个softmax转为概率, dim=-1的意思是将vocab_size维度进行归一化, 得到概率分布 [batch_size, vocab_size]
            probs = F.softmax(input=last_token_logits / temperature, dim=-1)

            # 这里没有使用argmax来拿概率最大的, 而是使用多项分布采样, 也就是说可能从概率最高的几个中进行选择
            # 这样配合 temperature, 来控制生成token的随机性
            # 例如 temperature 为1的时候, probs不受影响
            # 如果 temperature 为2的时候, 前面的在softmax前的logits / temperature, 这样大的值在softmax后, 原本和另一个token之间的概率差会很大, 这个时候经过前面的除法之后再softmax, 相差就不会像temperature为1的时候那么大, 那么另一个token就有可能被选到, 所以随机性增加了
            # 如果 temperature 为0.1的时候, 代表原本就很大的值在softmax之后, 占用的概率会更大, 这样就是大概率选择高概率的token, 实现更稳定的输出
            idx_next = torch.multinomial(input=probs, num_samples=1)  # [batch_size, 1]

            # 这个时候根据logits中最后一个token预测的概率分布, 找到最大的值所在vocab_size的下标, 也就是某个token在词汇表中的编码值
            # 通过vocab_dict[token_id]就找到了对应的token真正的样子. 代码就是encoder.decode([token_id])
            # 前面已经预测出了一个token_id, 这个时候需要拼接到前面的idx中, 继续补全, 直到达到max_new_token个数
            # 这里idx_next要插入到idx的第一个维度, idx是[batch_size, idx_length]
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def get_batch_x_y(data_set: str):
    data = train_data if data_set == "train" else val_data
    # 生成一个随机的下标, 注意这里size是元组, 代表我们每个批次选一个开始下标
    idxs = torch.randint(low=0, high=len(tokenized_text) - context_length, size=(batch_size,))
    # 这样就可以继续从tokenized_texta中取出对应批次的数据, x_batch是输入, y_batch是预期输出
    x_batch = torch.stack([tokenized_text[idx:idx + context_length] for idx in idxs])
    y_batch = torch.stack([tokenized_text[idx + 1:idx + context_length + 1] for idx in idxs])
    return x_batch, y_batch


model = TransformerLanguageModel()

losses = []

# 论文中使用的optimizer是Adam, 关于Adam\AdamW可以看论文: https://arxiv.org/pdf/1711.05101
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
for step in range(max_iters):
    x, y = get_batch_x_y("train")
    logits, loss = model(x, y)
    # 每次backward计算梯度都会放到.grad中, 如果不清空, 本次计算的梯度会加到上一次的梯度中. 所以这里是清空 param.grad，为训练做准备
    optimizer.zero_grad(set_to_none=True)
    # 计算梯度，并将其存储在 param.grad 里
    loss.backward()
    # 让优化器根据 param.grad 的值更新参数
    optimizer.step()

    losses.append(loss.item())  # 记录 loss

    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

# 绘制 loss 变化曲线
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.show()

torch.save(model.state_dict(), "model.pt")


model.eval()
start = "The salesperson"
start_idx = encoding.encode(start)
# [None, ...]的意思是将start_idx的维度从[idx_length] 变到 [1, idx_length], 添加上batch_size为1的维度
x = torch.tensor(start_idx, dtype=torch.long)[None, ...]
# out的维度是: [batch_size, idx_length + max_new_token]
out = model.generate(x, 1)

# out[0]代表取第一个batch
print(encoding.decode(out[0].tolist()))
