import os
import math
import numpy as np
from datautil.dataloader import load_data
from collections import Counter


def create_vocabs(path):
    wd_counter, tag_counter = Counter(), Counter()
    char_counter = Counter()
    insts = load_data(path)
    for inst in insts:
        wd_counter.update(inst.words)
        tag_counter.update(inst.tags)
        for wd in inst.words:
            char_counter.update(wd)

    return WordVocab(wd_counter, tag_counter), CharVocab(char_counter)


class WordVocab(object):
    def __init__(self, wd_counter, tag_counter):
        super(WordVocab, self).__init__()
        self.PAD = 0
        self.UNK = 1
        self._min_count = 5
        self._wd2freq = {wd: freq for wd, freq in wd_counter.items()}
        self._wd2idx = {'<pad>': self.PAD, '<unk>': self.UNK}

        for wd in self._wd2freq.keys():
            if wd not in self._wd2idx:
                self._wd2idx[wd] = len(self._wd2idx)

        self._idx2wd = {idx: wd for wd, idx in self._wd2idx.items()}

        # self._tag2idx = dict((tag, idx) for idx, tag in enumerate(tag_counter.keys()))
        self._tag2idx = {tag: idx for idx, tag in enumerate(tag_counter.keys())}
        self._idx2tag = {idx: tag for tag, idx in self._tag2idx.items()}
        print('tag number:', len(self._tag2idx))
        print('tags:', self._tag2idx.keys())

        self._extwd2idx = {'<pad>': self.PAD, '<unk>': self.UNK}
        self._extidx2wd = dict()

    # Reliability Signals
    # 只统计了训练预料中的词频，无法统计预训练词向量文件中的词频
    def build_signal_embed(self, corpus_func=lambda x: math.tanh(0.01*x)):
        signal_dim = 5
        signal_embed = np.zeros((self.ext_vocab_size, signal_dim), dtype=np.float32)

        for idx, wd in self._extidx2wd.items():
            signal_embed[idx, :] = np.asarray([
                corpus_func(self._wd2freq.get(wd, 0)),
                1 if self._wd2freq.get(wd, 0) < 5 else 0,  # 0 if it's oov according to word frequency
                1 if self._wd2freq.get(wd, 0) < 10 else 0,
                1 if self._wd2freq.get(wd, 0) < 50 else 0,
                1 if self._wd2freq.get(wd, 0) < 100 else 0
            ])

        return signal_embed

    def get_embedding_weights(self, embed_path):
        assert os.path.exists(embed_path)
        vec_size = 0
        vec_tabs = dict()
        with open(embed_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                tokens = line.strip().split()
                wd, vec = tokens[0], tokens[1:]
                if vec_size == 0:
                    vec_size = len(vec)
                vec_tabs[wd] = np.asarray(vec, np.float32)

        oov = 0
        for wd in self._wd2idx.keys():
            if wd not in vec_tabs:
                oov += 1

        print('oov ratio: %.3f%%' % (100 * (oov-2) / (len(self._wd2idx)-2)))

        for wd in vec_tabs.keys():
            if wd not in self._extwd2idx:
                self._extwd2idx[wd] = len(self._extwd2idx)

        self._extidx2wd = {idx: wd for wd, idx in self._extwd2idx.items()}

        vocab_size = len(self._extwd2idx)
        print('vocab size:', vocab_size)

        embedding_weights = np.zeros((vocab_size, vec_size), dtype=np.float32)
        for wd, idx in self._extwd2idx.items():
            if idx != self.PAD and idx != self.UNK:
                embedding_weights[idx] = vec_tabs[wd]
        
        embedding_weights[self.UNK] = np.mean(embedding_weights, 0) / np.std(embedding_weights)

        return embedding_weights

    def tag2index(self, tags):
        if isinstance(tags, list):
            return [self._tag2idx.get(t, -1) for t in tags]
        else:
            return self._tag2idx.get(tags, -1)

    def index2tag(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2tag.get(i, '<unk>') for i in idxs]
        else:
            return self._idx2tag.get(idxs, '<unk>')

    def word2index(self, wds):
        if isinstance(wds, list):
            return [self._wd2idx.get(wd, self.UNK) for wd in wds]
        else:
            return self._wd2idx.get(wds, self.UNK)

    def index2word(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2wd.get(i, '<unk>') for i in idxs]
        else:
            return self._idx2wd.get(idxs, '<unk>')

    def extword2index(self, wds):
        if isinstance(wds, list):
            return [self._extwd2idx.get(wd, self.UNK) for wd in wds]
        else:
            return self._extwd2idx.get(wds, self.UNK)

    def extindex2word(self, idxs):
        if isinstance(idxs, list):
            return [self._extidx2wd.get(i, '<unk>') for i in idxs]
        else:
            return self._extidx2wd.get(idxs, '<unk>')

    @property
    def vocab_size(self):
        return len(self._wd2idx)

    @property
    def ext_vocab_size(self):
        return len(self._extwd2idx)

    @property
    def tag_size(self):
        return len(self._tag2idx)


class CharVocab(object):
    def __init__(self, char_counter):
        super(CharVocab, self).__init__()
        self._UNK = 0  # 未知字符索引
        self._alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'
        char_counter.update(self._alphabet)

        # self._ch2idx = dict((ch, idx+1) for idx, ch in enumerate(char_counter.keys()))
        self._ch2idx = {ch: idx+1 for idx, ch in enumerate(char_counter.keys())}
        self._ch2idx['<unk>'] = self._UNK
        self._idx2ch = {idx: ch for ch, idx in self._ch2idx.items()}
        print('char vocab size:', len(self._ch2idx))

    def char2index(self, chs):
        if isinstance(chs, list) or len(chs) > 1:
            return [self._ch2idx.get(c, self._UNK) for c in chs]
        else:
            return self._ch2idx.get(chs, self._UNK)

    def index2char(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2ch.get(i, '<unk>') for i in idxs]
        else:
            return self._idx2ch.get(idxs, '<unk>')

    @property
    def vocab_size(self):
        return len(self._ch2idx)

