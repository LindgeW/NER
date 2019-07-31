import os
import numpy as np
import torch
import re
'''
IOB：O不属于实体，B表示实体名称的开头，I表示该实体的延续
实体：PER-人名  LOC-地点  ORG-组织  MISC-混合

O
B-PER
I-PER
B-LOC
I-LOC
B-ORG
I-ORG
B-MISC
I-MISC
'''


class Instance(object):
    def __init__(self, tokens=None, tags=None):
        self.words = tokens  # 词序列
        self.tags = tags  # 实体标签序列

    def __str__(self):
        return ' '.join(self.words) + ' _ ' + ' '.join(self.tags)


class Batch(object):
    def __init__(self, wd_src=None, ch_src=None, tgt=None, mask=None):
        self.wd_src = wd_src  # 词索引
        self.ch_src = ch_src  # 字符索引
        self.tgt = tgt        # 标签索引
        self.mask = mask      # 非填充部分的mask


'''
BIO -> BIOES (Begin Inside Outside End Single)
如：OBOBIIIO -> OSOBIIEO
'''
def bio2bioes(bio_tags):
    tag_len = len(bio_tags)
    for i, t in enumerate(bio_tags):
        if 'B-' in t and (i+1 == tag_len or 'I' not in bio_tags[i+1]):
            _type = bio_tags[i].split('-')[1]
            bio_tags[i] = 'S-' + _type
        elif 'I-' in t and (i+1 == tag_len or 'I-' not in bio_tags[i+1]):
            _type = bio_tags[i].split('-')[1]
            bio_tags[i] = 'E-' + _type

    return bio_tags


'''
一条标注数据的形式： [word][POS tag][chunk tag][NER tag]
'''
def load_data(path):
    def preprocess(src):
        return re.sub(r'\d+', '0', src.strip())

    assert os.path.exists(path)
    insts = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as fin:
        # 读取文件所有内容(字符串)
        all_content = fin.read()
        sections = all_content.strip().split('\n\n')
        for sec in sections:
            token_lines = sec.strip().split('\n')
            tokens, lbls = [], []
            for line in token_lines:
                line = preprocess(line)
                ts = line.split(' ')
                tokens.append(ts[0].strip())
                lbls.append(ts[-1].strip())  # BIO labels
            bioes_lbls = bio2bioes(lbls)  # BIO标签序列转BIOES标签序列
            insts.append(Instance(tokens, bioes_lbls))

    np.random.shuffle(insts)

    return insts


def batch_iter(dataset, batch_size, wd_vocab, ch_vocab, device=torch.device('cpu'), shuffle=True):
    if shuffle:
        np.random.shuffle(dataset)

    nb_batch = int(np.ceil(len(dataset) / batch_size))

    for i in range(nb_batch):
        batch_data = dataset[i*batch_size: (i+1)*batch_size]
        if shuffle:
            np.random.shuffle(batch_data)

        yield batch_gen(batch_data, wd_vocab, ch_vocab, device)


def batch_gen(batch_data, wd_vocab, ch_vocab, device=torch.device('cpu')):
    batch_size = len(batch_data)
    max_seq_len, max_ch_len = 0, 0
    for inst in batch_data:
        if max_seq_len < len(inst.words):
            max_seq_len = len(inst.words)
        for wd in inst.words:
            if max_ch_len < len(wd):
                max_ch_len = len(wd)

    wd_idxs = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to(device)
    tag_idxs = torch.zeros((batch_size, max_seq_len), dtype=torch.long).fill_(-1).to(device)
    ch_idxs = torch.zeros((batch_size, max_seq_len, max_ch_len), dtype=torch.long).to(device)
    non_pad_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.uint8).to(device)

    for i, inst in enumerate(batch_data):
        seq_len = len(inst.words)
        wd_idxs[i, :seq_len] = torch.tensor(wd_vocab.extword2index(inst.words))
        tag_idxs[i, :seq_len] = torch.tensor(wd_vocab.tag2index(inst.tags))
        non_pad_mask[i, :seq_len].fill_(1)

        for j, wd in enumerate(inst.words):
            ch_idxs[i, j, :len(wd)] = torch.tensor(ch_vocab.char2index(wd))

    return Batch(wd_idxs, ch_idxs, tag_idxs, non_pad_mask)
