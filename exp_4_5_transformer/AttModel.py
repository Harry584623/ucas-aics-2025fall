# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
from modules import *
from hyperparams import *
from hyperparams import Hyperparams as hp


class AttModel(nn.Module):
    def __init__(self, hp_, enc_voc, dec_voc):
        """Attention is all you need. https://arxiv.org/abs/1706.03762"""
        super(AttModel, self).__init__()
        self.hp = hp_
        self.enc_voc = enc_voc
        self.dec_voc = dec_voc

        # encoder embedding
        self.enc_emb = embedding(self.enc_voc, self.hp.hidden_units, zeros_pad=True, scale=True)
        print("Embedding PASS!")

        # encoder positional encoding
        if self.hp.sinusoid:
            self.enc_positional_encoding = positional_encoding(self.hp.hidden_units, zeros_pad=False, scale=False)
        else:
            self.enc_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, zeros_pad=False, scale=False)
        print("PositionEncoding PASS!")

        # dropout
        self.enc_dropout = nn.Dropout(self.hp.dropout_rate)

        for i in range(self.hp.num_blocks):
            self.__setattr__(
                'enc_self_attention_%d' % i,
                multihead_attention(self.hp, self.hp.hidden_units, self.hp.num_heads, self.hp.dropout_rate, causality=False)
            )
            self.__setattr__(
                'enc_feed_forward_%d' % i,
                feedforward(self.hp.hidden_units, [self.hp.hidden_units * 4, self.hp.hidden_units])
            )

        print("LayerNormalization PASS!")
        print("MutiheadAtt PASS!")
        print("FeedForward PASS!")

        # decoder embedding
        self.dec_emb = embedding(self.dec_voc, self.hp.hidden_units, zeros_pad=True, scale=True)

        # decoder positional encoding
        if self.hp.sinusoid:
            self.dec_positional_encoding = positional_encoding(self.hp.hidden_units, zeros_pad=False, scale=False)
        else:
            self.dec_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, zeros_pad=False, scale=False)

        # decoder dropout
        self.dec_dropout = nn.Dropout(self.hp.dropout_rate)

        for i in range(self.hp.num_blocks):
            self.__setattr__(
                'dec_self_attention_%d' % i,
                multihead_attention(self.hp, self.hp.hidden_units, self.hp.num_heads, self.hp.dropout_rate, causality=True)
            )
            self.__setattr__(
                'dec_vanilla_attention_%d' % i,
                multihead_attention(self.hp, self.hp.hidden_units, self.hp.num_heads, self.hp.dropout_rate, causality=False)
            )
            self.__setattr__(
                'dec_feed_forward_%d' % i,
                feedforward(self.hp.hidden_units, [self.hp.hidden_units * 4, self.hp.hidden_units])
            )

        self.logits_layer = nn.Linear(self.hp.hidden_units, self.dec_voc)
        self.label_smoothing = label_smoothing()
        print("LabelSmoothing PASS!")

    def forward(self, x, y):
        # decoder inputs: prepend <S> (id=2), shift right
        input_tensor = torch.ones(y[:, :1].size())
        if x.is_cuda:
            input_tensor = input_tensor.cuda()
        if x.device.type == 'mlu':
            input_tensor = input_tensor.to('mlu')  # TODO: mlu
        self.decoder_inputs = torch.cat([Variable(input_tensor * 2).long(), y[:, :-1]], dim=-1)

        # Encoder
        self.enc = self.enc_emb(x)
        if self.hp.sinusoid:
            self.enc += self.enc_positional_encoding(x, self.enc)
        else:
            enc_positional = torch.unsqueeze(torch.arange(0, x.size()[1]), 0).repeat(x.size(0), 1).long()
            if x.is_cuda:
                enc_positional = enc_positional.cuda()
            if x.device.type == 'mlu':
                enc_positional = enc_positional.to('mlu')
            self.enc += self.enc_positional_encoding(Variable(enc_positional))

        self.enc = self.enc_dropout(self.enc)

        for i in range(self.hp.num_blocks):
            self.enc = self.__getattr__('enc_self_attention_%d' % i)(self.enc, self.enc, self.enc)
            self.enc = self.__getattr__('enc_feed_forward_%d' % i)(self.enc)

        # Decoder
        self.dec = self.dec_emb(self.decoder_inputs)
        if self.hp.sinusoid:
            self.dec += self.dec_positional_encoding(self.decoder_inputs, self.dec)
        else:
            dec_positional = torch.unsqueeze(torch.arange(0, self.decoder_inputs.size()[1]), 0) \
                .repeat(self.decoder_inputs.size(0), 1).long()
            if x.is_cuda:
                dec_positional = dec_positional.cuda()
            if x.device.type == 'mlu':
                dec_positional = dec_positional.to('mlu')
            self.dec += self.dec_positional_encoding(Variable(dec_positional))

        self.dec = self.dec_dropout(self.dec)

        for i in range(self.hp.num_blocks):
            self.dec = self.__getattr__('dec_self_attention_%d' % i)(self.dec, self.dec, self.dec)
            self.dec = self.__getattr__('dec_vanilla_attention_%d' % i)(self.dec, self.enc, self.enc)
            self.dec = self.__getattr__('dec_feed_forward_%d' % i)(self.dec)

        # Final projection
        self.logits = self.logits_layer(self.dec)  # (B, T, V)

        # probs: flatten to (B*T, V) for loss
        probs_3d = F.softmax(self.logits, dim=-1)
        self.probs = probs_3d.view(-1, self.dec_voc)

        # preds: (B, T)
        _, preds_flat = torch.max(self.probs, dim=1)
        self.preds = preds_flat.view(y.size())

        self.istarget = (1. - y.eq(0.).float()).view(-1)
        self.acc = torch.sum(self.preds.eq(y).float().view(-1) * self.istarget) / torch.sum(self.istarget)

        # Loss
        self.y_onehot = torch.zeros(self.logits.size()[0] * self.logits.size()[1], self.dec_voc)
        if x.is_cuda:
            self.y_onehot = self.y_onehot.cuda()
        if x.device.type == 'mlu':
            self.y_onehot = self.y_onehot.to('mlu')

        self.y_onehot = Variable(self.y_onehot.scatter_(1, y.view(-1, 1).data, 1))
        self.y_smoothed = self.label_smoothing(self.y_onehot)

        self.loss = -torch.sum(self.y_smoothed * torch.log(self.probs + 1e-12), dim=-1)
        self.mean_loss = torch.sum(self.loss * self.istarget) / torch.sum(self.istarget)

        return self.mean_loss, self.preds, self.acc
