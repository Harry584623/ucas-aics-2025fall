# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
from torch.nn.parameter import Parameter
from hyperparams import *


class embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):
        """Embeds a given Variable.
        Args:
          vocab_size: Vocabulary size.
          num_units: Embedding hidden units.
          zeros_pad: if True, id 0 row is all zeros.
          scale: if True, outputs multiplied by sqrt(num_units).
        """
        super(embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = Parameter(torch.Tensor(vocab_size, num_units))
        # TODO：使用Xavier正态初始化方法对嵌入矩阵进行初始化
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        # TODO：调用内置函数实现 embedding
        outputs = F.embedding(
            inputs, self.lookup_table, self.padding_idx, None, 2, False, False
        )

        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class layer_normalization(nn.Module):
    def __init__(self, features, epsilon=1e-8):
        """Applies layer normalization."""
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        # TODO：gamma 初始全1
        self.gamma = Parameter(torch.ones(features))
        # TODO：beta 初始全0
        self.beta = Parameter(torch.zeros(features))

    def forward(self, x):
        # TODO：最后一维均值
        mean = torch.mean(x, dim=-1, keepdim=True)
        # TODO：最后一维标准差
        std = torch.std(x, dim=-1, keepdim=True)
        # TODO：层归一化
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


class positional_encoding(nn.Module):
    def __init__(self, num_units, zeros_pad=True, scale=True):
        """Sinusoidal Positional Encoding."""
        super(positional_encoding, self).__init__()
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale

    def forward(self, inputs, y):
        # inputs: (N, T)
        N, T = inputs.size()[0: 2]

        position_ind = torch.unsqueeze(torch.arange(0, T), 0).repeat(N, 1)
        if inputs.is_cuda:
            position_ind = Variable(position_ind.cuda().long())
        if inputs.device.type == 'mlu':
            position_ind = Variable(position_ind.to('mlu').long())

        # TODO: 位置编码矩阵 (T, num_units)
        # position_enc[pos, i] = pos / 10000^(2i/num_units)
        position_enc = np.array(
            [
                [pos / np.power(10000, 2.0 * (i // 2) / self.num_units) for i in range(self.num_units)]
                for pos in range(T)
            ],
            dtype=np.float32
        )
        position_enc = torch.Tensor(position_enc)

        # TODO：偶数列 sin
        position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])
        # TODO：奇数列 cos
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])

        lookup_table = Variable(position_enc)
        if inputs.is_cuda:
            lookup_table = lookup_table.cuda()
        if inputs.device.type == 'mlu':
            lookup_table = lookup_table.to('mlu').to(y.dtype)

        if self.zeros_pad:
            lookup_table = torch.cat((Variable(torch.zeros(1, self.num_units)), lookup_table[1:, :]), 0)
            padding_idx = 0
        else:
            padding_idx = -1

        # TODO：根据 position_ind 查表取位置向量
        outputs = F.embedding(
            position_ind, lookup_table, padding_idx, None, 2, False, False
        )

        if self.scale:
            # TODO：缩放
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class multihead_attention(nn.Module):
    def __init__(self, hp_, num_units, num_heads=8, dropout_rate=0, causality=False):
        """Applies multihead attention."""
        super(multihead_attention, self).__init__()
        self.hp = hp_
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality

        # TODO：Q/K/V 线性映射 + ReLU
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        # TODO：输出 dropout 层
        self.output_dropout = nn.Dropout(self.dropout_rate)
        # TODO：层归一化
        self.normalization = layer_normalization(self.num_units)

    def forward(self, queries, keys, values):
        # queries: (N, T_q, C), keys/values: (N, T_k, C)
        Q = self.Q_proj(queries)
        K = self.K_proj(keys)
        V = self.V_proj(values)

        # TODO：按 head 切分并在 batch 维拼接
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)

        # TODO：注意力得分 (h*N, T_q, T_k)
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))

        # TODO：缩放
        outputs = outputs / (K_.size(-1) ** 0.5)

        # Key masking
        key_masks = torch.sign(torch.abs(torch.sum(keys, dim=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(self.num_heads, 1)            # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)

        init_tensor = torch.ones(*outputs.size(), dtype=queries.dtype)
        if queries.is_cuda:
            init_tensor = init_tensor.cuda()
        if queries.device.type == 'mlu':
            init_tensor = init_tensor.to('mlu')
        padding = Variable(init_tensor * (-2 ** 32 + 1))

        condition = key_masks.eq(0.)
        outputs = padding * condition + outputs * (~condition)

        # Causality masking (future blinding)
        if self.causality:
            diag_vals = torch.ones(*outputs[0, :, :].size(), dtype=queries.dtype)  # (T_q, T_k)
            if queries.is_cuda:
                diag_vals = diag_vals.cuda()
            if queries.device.type == 'mlu':
                diag_vals = diag_vals.to('mlu')

            # TODO：下三角
            tril = torch.tril(diag_vals)
            masks = Variable(torch.unsqueeze(tril, 0).repeat(outputs.size()[0], 1, 1))

            mask = torch.ones(*masks.size(), dtype=queries.dtype)
            if queries.is_cuda:
                mask = mask.cuda()
            if queries.device.type == 'mlu':
                mask = mask.to('mlu')
            # TODO：下三角（重复一处，按讲义结构保留）
            tril = torch.tril(diag_vals)
            padding = Variable(mask * (-2 ** 32 + 1))

            condition = masks.eq(0.)
            outputs = padding * condition + outputs * (~condition)

        # TODO：softmax
        outputs = F.softmax(outputs, dim=-1)

        # Query masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)              # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        # TODO：屏蔽无效 query
        outputs = outputs * query_masks

        # TODO：dropout
        outputs = self.output_dropout(outputs)

        # TODO：加权求和
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)

        # TODO：拼回 (N, T_q, C)
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)

        # TODO：残差
        outputs += queries

        # TODO：层归一化
        outputs = self.normalization(outputs)
        return outputs


class feedforward(nn.Module):
    def __init__(self, in_channels, num_units=[2048, 512]):
        super(feedforward, self).__init__()
        self.in_channels = in_channels
        self.num_units = num_units

        self.conv = False
        if self.conv:
            params = {'in_channels': self.in_channels, 'out_channels': self.num_units[0],
                      'kernel_size': 1, 'stride': 1, 'bias': True}
            self.conv1 = nn.Sequential(nn.Conv1d(**params), nn.ReLU())
            params = {'in_channels': self.num_units[0], 'out_channels': self.num_units[1],
                      'kernel_size': 1, 'stride': 1, 'bias': True}
            self.conv2 = nn.Conv1d(**params)
        else:
            # 关键：训练时的结构是 Sequential(Linear, ReLU)，这样 state_dict 才会是 conv1.0.weight
            self.conv1 = nn.Sequential(
                nn.Linear(self.in_channels, self.num_units[0]),
                nn.ReLU()
            )
            self.conv2 = nn.Linear(self.num_units[0], self.num_units[1])

        self.normalization = layer_normalization(self.in_channels)

    def forward(self, inputs):
        residual = inputs
        if self.conv:
            inputs = inputs.transpose(1, 2)
            residual = inputs
            outputs = self.conv1(inputs)
            outputs = self.conv2(outputs)
            outputs += residual
            outputs = outputs.transpose(1, 2)
            outputs = self.normalization(outputs)
        else:
            # 关键：这里不要再额外 F.relu，因为 conv1 已经带 ReLU 了
            outputs = self.conv1(inputs)
            outputs = self.conv2(outputs)
            outputs += residual
            outputs = self.normalization(outputs)
        return outputs


class label_smoothing(nn.Module):
    def __init__(self, epsilon=0.1):
        """Label smoothing."""
        super(label_smoothing, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs):
        # inputs: (N, K)
        # TODO：类别数
        K = inputs.size(-1)
        # TODO：label smoothing
        return (1 - self.epsilon) * inputs + self.epsilon / K
