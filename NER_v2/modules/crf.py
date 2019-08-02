import torch
import torch.nn as nn


# CRF with batch and mask
class BatchCRF(nn.Module):
    def __init__(self, nb_tags, batch_first=False):
        super(BatchCRF, self).__init__()

        self._batch_first = batch_first

        # CRF层参数
        self._transitions = nn.Parameter(torch.empty(nb_tags, nb_tags))  # (nb_tags, nb_tags)
        self._start_trans = nn.Parameter(torch.empty(nb_tags))  # (nb_tags, )
        self._end_trans = nn.Parameter(torch.empty(nb_tags))  # (nb_tags, )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self._transitions, -0.1, 0.1)
        nn.init.uniform_(self._start_trans, -0.1, 0.1)
        nn.init.uniform_(self._end_trans, -0.1, 0.1)
        # nn.init.normal_(self._transitions)
        # nn.init.normal_(self._start_trans)
        # nn.init.normal_(self._end_trans)

    # 根据给定的发射概率和真实的标签序列，计算负对数似然值(误差)
    def neg_likelihood(self, emissions, tags, mask=None):
        '''
        :param emissions: (seq_len, batch_size, nb_tags)
        :param tags: (seq_len, batch_size)
        :param mask: (seq_len, batch_size)
        :return: loss tensor
        '''

        if mask is None:
            # 根据tags的形状创建单位矩阵（表明所有元素都有效）
            mask = tags.new_ones(tags.shape).byte()

        if self._batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # 最优路径得分
        gold_score = self._gold_score(emissions, tags, mask)
        # 所有可能路径得分
        all_score = self._all_path_score(emissions, mask)
        # (batch_size, )
        nlld = all_score - gold_score  # 对应于CRF的最大熵公式
        return nlld.mean()
        # return nlld.sum() / mask.sum().item()

    # 根据实际标签序列，计算最优路径分数
    def _gold_score(self, emissions, tags, mask=None):
        '''
        :param emissions: (seq_len, batch_size, nb_tags)  发射概率
        :param tags: (seq_len, batch_size)  标签序列
        :param mask: (seq_len, batch_size)  byte tensor
        :return: (batch_size, )  loss tensor
        '''
        seq_len, batch_size, _ = emissions.size()
        mask = mask.float()  # byte类型不能参与运算
        # (batch_size, )
        score = self._start_trans[tags[0]] + emissions[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_len):
            # 原始score + 转移score + 当前标签对应的发射score
            score += self._transitions[tags[i-1], tags[i]] * mask[i] \
                     + emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # 序列实际长度: (batch_size, )
        # 注：由于不同序列的长度不一致，最后一步的转移需要根据实际长度求
        len_mask = mask.long().sum(dim=0) - 1
        # last_tag = tags[len_mask, :batch_size]
        # (batch_size, )
        last_tags = tags[len_mask, torch.arange(batch_size)]
        score += self._end_trans[last_tags]

        return score

    # 计算所有可能的路径
    def _all_path_score(self, emissions, mask=None):
        '''
        :param emissions: (seq_len, batch_size, nb_tags)
        :param mask: (seq_len, batch_size)
        :return: (batch_size, )
        '''
        seq_len = emissions.size(0)
        # (batch_size, nb_tags)
        score = self._start_trans + emissions[0]
        for i in range(1, seq_len):
            # (batch_size, nb_tags, 1) + (nb_tags, nb_tags) + (batch_size, 1, nb_tags)
            # (batch_size, nb_tags, nb_tags)
            next_score = score.unsqueeze(-1) + self._transitions + emissions[i].unsqueeze(1)
            # (batch_size, nb_tags)
            next_score = torch.logsumexp(next_score, dim=1)
            # 对无效的pad部分，保留原始的score
            # (batch_size, nb_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self._end_trans
        # (batch_size, )
        return torch.logsumexp(score, dim=1)

    # 解码，用于预测时，找到最优路径（标签序列）
    def decode(self, emissions, mask=None):
        '''
        :param emissions: (seq_len, batch_size, nb_tags)
        :param mask: (seq_len, batch_size)
        :return: (batch_size, seq_len)  tag indices list
        '''
        if mask is None:
            # 根据emissions前两维创建单位矩阵mask，表明所有元素都有效
            mask = emissions.new_ones(emissions.shape[:2]).byte()

        if self._batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        seq_len, batch_size, _ = emissions.size()

        hist_path = []
        # (batch_size, nb_tags)
        score = self._start_trans + emissions[0]
        for i in range(1, seq_len):
            # (batch_size, nb_tags, 1) + (nb_tags, nb_tags) + (batch_size, 1, nb_tags)
            # (batch_size, nb_tags, nb_tags)
            next_score = score.unsqueeze(-1) + self._transitions + emissions[i].unsqueeze(1)
            # (batch_size, nb_tags)
            next_score, idxs = next_score.max(dim=1)
            hist_path.append(idxs)  # seq_len个(batch_size, nb_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self._end_trans

        # (batch_size, )
        len_mask = mask.long().sum(dim=0) - 1

        best_paths = []
        # 回溯法，根据最大score逆向求解最优序列路径
        for i in range(batch_size):
            best_last_tag = score[i].argmax(dim=0)
            best_tags = [best_last_tag.item()]
            for hist in reversed(hist_path[:len_mask[i]]):  # 根据序列实际长度
                best_last_tag = hist[i][best_tags[-1]]
                best_tags.append(best_last_tag.item())
            best_tags.reverse()
            best_paths.append(best_tags)

        return best_paths

    def forward(self, emissions, tags, mask=None):
        return self.neg_likelihood(emissions, tags, mask)
