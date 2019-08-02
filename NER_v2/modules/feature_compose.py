import torch
import torch.nn as nn


class EmbedCompose(nn.Module):
    def __init__(self, wd_embed_dim, sig_embed_dim=5):
        super(EmbedCompose, self).__init__()
        self._wd_gates = nn.ModuleList([
            nn.Linear(in_features=wd_embed_dim, out_features=wd_embed_dim),
            nn.Linear(in_features=wd_embed_dim, out_features=wd_embed_dim)
        ])

        self._ch_gates = nn.ModuleList([
            nn.Linear(in_features=wd_embed_dim, out_features=wd_embed_dim),
            nn.Linear(in_features=wd_embed_dim, out_features=wd_embed_dim)
        ])

        self._sig_gates = nn.ModuleList([
            nn.Linear(in_features=sig_embed_dim, out_features=wd_embed_dim),
            nn.Linear(in_features=sig_embed_dim, out_features=wd_embed_dim)
        ])

    def forward(self, wd_embed, ch_embed, sig_embed):
        '''
        :param wd_embed: (batch_size, seq_len, wd_embed_dim)
        :param ch_embed: (batch_size, seq_len, ch_embed_dim)
        :param sig_embed: (batch_size, seq_len, sig_embed_dim)
        :return:
        '''
        wd_embed_dim = wd_embed.size(-1)
        ch_embed_dim = ch_embed.size(-1)
        assert ch_embed_dim > wd_embed_dim

        # left: (batch_size, seq_len, wd_embed_dim)
        # right: (batch_size, seq_len, ch_embed_dim - wd_embed_dim)
        ch_embed_left, ch_embed_right = ch_embed.split(wd_embed_dim, dim=-1)

        # (batch_size, seq_len, wd_embed_dim)
        gate_w = (self._wd_gates[0](wd_embed) +
                  self._ch_gates[0](ch_embed_left) +
                  self._sig_gates[0](sig_embed)).sigmoid()

        gate_c = (self._wd_gates[1](wd_embed) +
                  self._ch_gates[1](ch_embed_left) +
                  self._sig_gates[1](sig_embed)).sigmoid()

        embed = gate_w * wd_embed + gate_c * ch_embed_left

        # (batch_size, seq_len, ch_embed_dim)
        return torch.cat((embed, ch_embed_right), dim=-1)


class FeatureCompose(nn.Module):
    def __init__(self, hidden_size, sig_dim, context_size=4, batch_first=True):
        super(FeatureCompose, self).__init__()

        self._ctx_size = context_size
        self._sig_dim = sig_dim
        self._batch_first = batch_first
        self._hidden_size = hidden_size

        self._fw_cof_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self._bw_cof_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size)

        self._hs_gates = nn.ModuleList([
            nn.Linear(in_features=hidden_size, out_features=hidden_size)
            for _ in range(4)
        ])

        self._cof_gates = nn.ModuleList([
            nn.Linear(in_features=hidden_size, out_features=hidden_size)
            for _ in range(4)
        ])

        self._rs_gates = nn.ModuleList([
            nn.Linear(in_features=(self._ctx_size+1)*sig_dim, out_features=hidden_size)
            for _ in range(4)
        ])

    def _feat_gate(self, hs, cof, rs, idx=None):
        '''
        :param hs: hidden state of rnn
        :param cof: context-only features
        :param rs:  reliability signals
        :param idx:
        :return:
        '''
        gate_ = self._hs_gates[idx](hs) + self._cof_gates[idx](cof) + self._rs_gates[idx](rs)
        return gate_.sigmoid()

    def forward(self, rnn_hidden, sig_inputs):
        '''
        :param rnn_hidden: (batch_size, seq_len, 2*hidden_size)
        :param sig_inputs: (batch_size, seq_len, sig_embed_dim)
        :return:
        '''
        batch_size = rnn_hidden.size(0)

        # (batch_size, seq_len, hidden_size)
        rnn_fw, rnn_bw = rnn_hidden.split(self._hidden_size, dim=-1)
        sig_pad = sig_inputs.new_zeros((batch_size, self._ctx_size, self._sig_dim), requires_grad=False)
        # (batch_size, seq_len+2*ctx_size, sig_dim)
        # -> (batch_size, (seq_len+2*ctx_size)*sig_dim)
        sig_inputs = torch.cat((sig_pad, sig_inputs, sig_pad), dim=1).reshape((batch_size, -1))
        # 生成(C+1)*sig_dim大小的窗口，每次滑动的距离为sig_dim
        sig_unfolds = sig_inputs.unfold(dimension=1,
                                        size=(self._ctx_size + 1) * self._sig_dim,
                                        step=self._sig_dim)
        # (batch_size, seq_len, (self._ctx_size + 1) * sig_dim)
        sig_fw = sig_unfolds[:, :-self._ctx_size, :]
        sig_bw = sig_unfolds[:, self._ctx_size:, :]
        # 生成h_0 和 h_n+1
        rnn_pad = rnn_fw.new_zeros((batch_size, 1, self._hidden_size), requires_grad=False)
        cof_fw = torch.cat((rnn_pad, rnn_fw), dim=1)[:, :-1, :]
        cof_bw = torch.cat((rnn_bw, rnn_pad), dim=1)[:, 1:, :]
        # (batch_size, seq_len, hidden_size)
        cof_fw = self._fw_cof_linear(cof_fw).tanh()
        cof_bw = self._bw_cof_linear(cof_bw).tanh()

        fw_gate_o = self._feat_gate(rnn_fw, cof_fw, sig_fw, 0)
        fw_gate_h = self._feat_gate(rnn_fw, cof_fw, sig_fw, 1)
        bw_gate_o = self._feat_gate(rnn_bw, cof_bw, sig_bw, 2)
        bw_gate_h = self._feat_gate(rnn_bw, cof_bw, sig_bw, 3)

        h_fw = fw_gate_o * cof_fw + fw_gate_h * rnn_fw
        h_bw = bw_gate_o * cof_bw + bw_gate_h * rnn_bw

        # (batch_size, seq_len, 2*hidden_size)
        return torch.cat((h_fw, h_bw), dim=-1)



