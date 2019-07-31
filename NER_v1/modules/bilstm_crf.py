import torch
import torch.nn as nn
from .attention import SelfAttention
from .char_embed import CharEmbedding
from .rnn_encoder import RNNEncoder
from .crf import BatchCRF
'''
    wd_idxs -> word embedding --> BiLSTM -> CRF -> tag sequence 
    ch_idxs -> char embedding |
    
    BOEMS/BOIES实体只有这三种形式：[b,m….,e]、[b,e]、 [s]，
    实体是有边界的，在精确匹配过程中，需要预测实体类别以及边界范围，
    只有这两部分都匹配成功，才算预测正确，否则是预测错误。
    
    实体这个概念可以很广，只要是业务需要的特殊文本片段都可以称为实体
'''


class BiLSTMCRF(nn.Module):
    def __init__(self, args, embedding_weights):
        super(BiLSTMCRF, self).__init__()
        self._pad = args.pad

        wd_embed_dim = embedding_weights.shape[1]
        self._wd_embed = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weights))
        self._wd_embed.weight.requires_grad = False

        self._char_embed = CharEmbedding(char_vocab_size=args.char_vocab_size,
                                         char_feature_size=args.char_feature_size,
                                         char_embed_dim=args.char_embed_dim)

        self._batch_first = True
        self._bidirectional = True
        nb_directions = 2 if self._bidirectional else 1

        self._bilstm = RNNEncoder(input_size=wd_embed_dim + args.char_embed_dim,
                                  hidden_size=args.hidden_size,
                                  num_layers=args.num_layers,
                                  bidirectional=self._bidirectional,
                                  batch_first=self._batch_first,
                                  rnn_type='lstm')

        self._linear = nn.Linear(in_features=args.hidden_size * nb_directions,
                                 out_features=args.tag_size)

        self._bcrf = BatchCRF(args.tag_size, self._batch_first)

        self._attention = SelfAttention()

        self._drop_embed = nn.Dropout(args.dropout_embed)
        self._drop_linear = nn.Dropout(args.dropout_linear)

    def _get_emissions(self, wd_input, ch_input, non_pad_mask):
        '''
        :param wd_input: (batch_size, seq_len)  词索引
        :param ch_input: (batch_size, seq_len, ch_len)  字符索引
        :param non_pad_mask: (batch_size, seq_len)  byte tensor
        :return:
        '''
        # (batch_size, seq_len, wd_embed_dim)
        wd_embed = self._wd_embed(wd_input)
        # (batch_size, ch_len, ch_embed_dim)
        ch_embed = self._char_embed(ch_input)
        embed = torch.cat((wd_embed, ch_embed), dim=-1)

        if self.training:
            embed = self._drop_embed(embed)

        # rnn_out: (batch_size, seq_len, hidden_size*nb_directions)
        # hidden: hn cn (num_layers, batch_size, hidden_size*nb_directions)
        rnn_out, _ = self._bilstm(embed, non_pad_mask)

        # seq_len = rnn_out.size(1)
        # # (batch_size, k_len) -> (batch_size, 1, k_len) -> (batch_size, q_len, k_len)
        # att_mask = non_pad_mask.unsqueeze(1).repeat(1, seq_len, 1)
        # # att_mask = non_pad_mask.unsqueeze(1).expand(-1, seq_len, -1)
        # # (batch_size, seq_len, hidden_size*nb_directions)
        # att_out = self._attention(rnn_out, rnn_out, rnn_out, att_mask)

        if self.training:
            rnn_out = self._drop_linear(rnn_out)

        # (batch_size, seq_len, tag_size)
        emissions = self._linear(rnn_out)  # 非归一化score

        return emissions
        # return F.softmax(emissions, dim=-1)  # 归一化score

    # 计算损失值
    def calc_loss(self, wd_input, ch_input, tags, non_pad_mask=None):
        '''
        :param wd_input: (batch_size, seq_len)  词索引
        :param ch_input: (batch_size, seq_len, ch_len)  字符索引
        :param tags: (batch_size, seq_len)  标签索引
        :param non_pad_mask: (batch_size, seq_len)  非填充部分mask
        :return: loss tensor
        '''
        # (batch_size, seq_len)
        if non_pad_mask is None:
            non_pad_mask = wd_input.ne(self._pad)  # 非运算求mask

        emissions = self._get_emissions(wd_input, ch_input, non_pad_mask)

        loss = self._bcrf(emissions, tags, non_pad_mask)

        return loss

    def forward(self, wd_input, ch_input, non_pad_mask=None):
        '''
        :param wd_input: (batch_size, seq_len)  词索引
        :param ch_input: (batch_size, seq_len, ch_len)  字符索引
        :param non_pad_mask: (batch_size, seq_len)  非填充部分mask
        :return: (batch_size, seq_len)  best tag sequences
        '''

        # (batch_size, seq_len)
        if non_pad_mask is None:
            non_pad_mask = wd_input.ne(self._pad)  # 非运算求mask

        emissions = self._get_emissions(wd_input, ch_input, non_pad_mask)

        tags_seq = self._bcrf.decode(emissions)

        return torch.tensor(tags_seq, wd_input.device)

