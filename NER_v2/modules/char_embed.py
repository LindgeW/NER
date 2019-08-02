import torch
import torch.nn as nn


class CharEmbedding(nn.Module):
    def __init__(self, char_vocab_size,
                 # filter_size,
                 char_feature_size,
                 char_embed_dim,
                 dropout=0.5):
        '''
        :param char_vocab_size: 字符表大小
        :param char_embed_dim: 字符向量大小
        '''
        super(CharEmbedding, self).__init__()
        self._ch_embedding = nn.Embedding(num_embeddings=char_vocab_size,
                                          embedding_dim=char_feature_size,
                                          padding_idx=0)
        # 激活函数tanh
        nn.init.xavier_uniform_(self._ch_embedding.weight)
        # 激活函数relu
        # nn.init.kaiming_uniform_(self._ch_embedding.weight)

        # 卷积层
        # 2-gram 3-gram 4-gram特征提取
        self._win_size = [2, 3, 4]
        self._convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=char_feature_size,  # 字符向量维度
                    # out_channels=filter_size,  # 卷积核的个数(固定)
                    out_channels=25*w,  # 卷积核的个数(可变)
                    kernel_size=w,  # 窗口大小
                    padding=1,   # 数据填充
                ),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(output_size=1)  # 自适应池化层
            ) for w in self._win_size])

        self._linear = nn.Linear(in_features=sum(self._win_size)*25, out_features=char_embed_dim)  # 字符嵌入的维度
        # self._linear = nn.Linear(in_features=len(self._win_size) * filter_size, out_features=char_embed_dim)  # 字符嵌入的维度

        self._drop_embed = nn.Dropout(dropout)
        self._drop_linear = nn.Dropout(dropout)

    def forward(self, ch_idxs):
        '''
        :param ch_idxs: [batch_size, max_seq_len, max_ch_len]
        :return:
        '''
        # [batch_size * max_seq_len, max_ch_len, char_embed_dim]
        batch_size, max_seq_len, max_ch_len = ch_idxs.size()
        ch_idxs = ch_idxs.reshape(-1, max_ch_len)
        embed = self._ch_embedding(ch_idxs)

        if self.training:
            embed = self._drop_embed(embed)

        # [batch_size * max_seq_len, max_ch_len, char_embed_dim]
        # -> [batch_size * max_seq_len, conv_out, max_ch_len]
        # -> [batch_size * max_seq_len, conv_out, 1]
        # -> [batch_size * max_seq_len, conv_out]
        embed = embed.transpose(1, 2)
        convs_out = torch.cat(tuple(convs(embed) for convs in self._convs), dim=1).squeeze(dim=2)

        if self.training:
            convs_out = self._drop_linear(convs_out)

        # [batch_size * max_seq_len,  conv_out]
        # -> [batch_size * max_seq_len, char_hidden_size]
        out = self._linear(convs_out)

        # [batch_size, max_seq_len, char_hidden_size]
        out = out.reshape(batch_size, max_seq_len, -1)

        return out
