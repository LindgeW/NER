import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, dropout=0.5):
        super(SelfAttention, self).__init__()
        self._inf = -1000
        self._dropout = dropout

    def forward(self, q, k, v, non_pad_mask=None):
        '''
        :param q: (batch_size, q_len, Q)
        :param k: (batch_size, k_len, K)
        :param v: (batch_size, v_len, V)
        Q = K = V, q_len = k_len
        :param non_pad_mask: (batch_size, q_len, k_len) 非填充部分mask
        :return: (batch_size, q_len, V)
        '''

        # (batch_size, q_len, Q) * (batch_size, K, k_len)
        # -> (batch_size, q_len, k_len)
        att_weights = torch.bmm(q, k.transpose(1, 2))
        att_weights /= math.sqrt(k.size(-1))

        if non_pad_mask is not None:
            att_weights.masked_fill_(non_pad_mask, self._inf)

        # (batch_size, q_len, k_len)
        soft_att_weights = F.softmax(att_weights, dim=-1)

        soft_att_weights = F.dropout(soft_att_weights, p=self._dropout, training=self.training)

        # (batch_size, q_len, k_len) * (batch_size, v_len, V)
        # -> (batch_size, q_len, V)
        att_out = torch.bmm(soft_att_weights, v)

        return att_out
