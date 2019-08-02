import torch
import torch.nn as nn
import torch.optim as optim
from datautil.dataloader import batch_iter
import time


class SequenceTagger(object):
    def __init__(self, model, args, wd_vocab, ch_vocab):
        assert isinstance(model, nn.Module)
        self._model = model
        print(next(self._model.parameters()).is_cuda)  # 判断模型是否在GPU上
        self._args = args
        self._wd_vocab = wd_vocab
        self._ch_vocab = ch_vocab

    def summary(self):
        print(self._model)

    def train(self, train_data, dev_data):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()),
                               lr=self._args.lr)

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        for ep in range(self._args.epoch):
            self._model.train()
            start = time.time()
            train_loss = 0
            nb_correct, nb_gold, nb_pred = 0, 0, 0

            lr_scheduler.step()
            print(lr_scheduler.get_lr())
            for batch in batch_iter(train_data, self._args.batch_size, self._wd_vocab, self._ch_vocab, device=self._args.device):
                self._model.zero_grad()
                loss = self._model.calc_loss(batch.wd_src, batch.ch_src, batch.tgt)
                # GPU并行计算，自定义方法无法并行(在主GPU中运行)，需要加.module调用
                # loss = self._model.module.calc_loss(batch.wd_src, batch.ch_src, batch.tgt)
                train_loss += loss.data.item()
                loss.backward()
                # 梯度裁剪 - 解决梯度爆炸问题
                nn.utils.clip_grad_value_(filter(lambda p: p.requires_grad, self._model.parameters()),
                                          clip_value=5.0)  # 限制梯度值在[-5, 5]之间

                optimizer.step()

                pred = self._model(batch.wd_src, batch.ch_src)
                result = self._calc_acc(pred, batch.tgt, batch.mask)
                nb_correct += result[0]
                nb_gold += result[1]
                nb_pred += result[2]

            train_f1 = self._calc_f1(nb_correct, nb_gold, nb_pred)
            print('[Epoch %d] train_loss: %.3f train_F1: %.3f' % (ep, train_loss, train_f1))

            dev_loss, dev_f1 = self._validate(dev_data)
            end = time.time()
            print('dev_loss: %.3f dev_F1: %.3f' % (dev_loss, dev_f1))
            print('time cost: %.2f s' % (end-start))

    def _validate(self, dev_data):
        dev_loss = 0
        nb_correct, nb_gold, nb_pred = 0, 0, 0
        self._model.eval()
        with torch.no_grad():
            for batch in batch_iter(dev_data, self._args.batch_size, self._wd_vocab, self._ch_vocab,
                                    device=self._args.device):
                loss = self._model.calc_loss(batch.wd_src, batch.ch_src, batch.tgt)
                dev_loss += loss.data.item()

                pred = self._model(batch.wd_src, batch.ch_src)
                result = self._calc_acc(pred, batch.tgt, batch.mask)
                nb_correct += result[0]
                nb_gold += result[1]
                nb_pred += result[2]

        return dev_loss, self._calc_f1(nb_correct, nb_gold, nb_pred)

    def evaluate(self, test_data):
        vail_loss = 0
        nb_correct, nb_gold, nb_pred = 0, 0, 0
        self._model.eval()
        with torch.no_grad():
            for batch in batch_iter(test_data, self._args.batch_size, self._wd_vocab, self._ch_vocab,
                                    device=self._args.device):
                loss = self._model.calc_loss(batch.wd_src, batch.ch_src, batch.tgt)
                vail_loss += loss.data.item()

                pred = self._model(batch.wd_src, batch.ch_src)
                result = self._calc_acc(pred, batch.tgt, batch.mask)
                nb_correct += result[0]
                nb_gold += result[1]
                nb_pred += result[2]

        test_f1 = self._calc_f1(nb_correct, nb_gold, nb_pred)
        print('======== test_loss: %.3f test_F1: %.3f ========' % (vail_loss, test_f1))
        return vail_loss, test_f1

    # IBOES-精确匹配
    def _exact_match(self, pred, target):
        if torch.is_tensor(pred) and torch.is_tensor(target):
            pred = pred.tolist()
            target = target.tolist()

        nb_correct, nb_gold, nb_pred = 0, 0, 0

        # 统计pred tags中总的实体数
        entity_type = None
        valid = False
        for p in pred:
            _type = self._wd_vocab.index2tag(p)
            if 'S-' in _type:
                nb_pred += 1
                valid = False
            elif 'B-' in _type:
                entity_type = _type.split('-')[1]
                valid = True
            elif 'I-' in _type:
                if entity_type != _type.split('-')[1]:
                    valid = False
            elif 'E-' in _type:
                if entity_type == _type.split('-')[1] and valid:
                    nb_pred += 1
                valid = False

        # 统计gold tags中总实体数以及预测正确的实体数
        begin = False
        for i, (t, p) in enumerate(zip(target, pred)):
            _type = self._wd_vocab.index2tag(t)
            if 'S-' in _type:
                nb_gold += 1
                if t == p:
                    nb_correct += 1
            elif 'B-' in _type:
                if t == p:
                    begin = True
            elif 'I-' in _type:
                if t != p:
                    begin = False
            elif 'E-' in _type:
                nb_gold += 1
                if t == p and begin:
                    nb_correct += 1

        return nb_correct, nb_gold, nb_pred

    # # IBO - 不完全匹配
    # def _overlap_match(self, pred, target):
    #     if torch.is_tensor(pred) and torch.is_tensor(target):
    #         pred = pred.tolist()
    #         target = target.tolist()
    #
    #     nb_correct, nb_gold, nb_pred = 0, 0, 0
    #     n = len(target)
    #
    #     # OBI - 比较边缘和实体类型
    #     for i, (t, p) in enumerate(zip(target, pred)):
    #         if 'B-' in self._wd_vocab.index2tag(p):
    #             nb_pred += 1
    #
    #         _type = self._wd_vocab.index2tag(t)
    #         if 'B-' in _type:
    #             nb_gold += 1
    #             if t == p and (i+1 == n or 'I-' not in self._wd_vocab.index2tag(target[i+1])):
    #                 nb_correct += 1
    #         elif 'I-' in _type:
    #             if t == p and (i+1 == n or 'O' == self._wd_vocab.index2tag(target[i+1])):
    #                 nb_correct += 1
    #
    #     return nb_correct, nb_gold, nb_pred
    #
    # # IBOES - 精确匹配
    # def _exact_match(self, pred, target):
    #     if torch.is_tensor(pred) and torch.is_tensor(target):
    #         pred = pred.tolist()
    #         target = target.tolist()
    #
    #     nb_correct, nb_gold, nb_pred = 0, 0, 0
    #     n = len(target)
    #
    #     begin = False
    #     for i, (t, p) in enumerate(zip(target, pred)):
    #         if 'B-' in self._wd_vocab.index2tag(p):
    #             nb_pred += 1
    #
    #         _type = self._wd_vocab.index2tag(t)
    #         if 'B-' in _type:
    #             nb_gold += 1
    #             if t == p:
    #                 if i+1 == n or 'I-' not in self._wd_vocab.index2tag(target[i+1]):
    #                     nb_correct += 1
    #                 else:  # B后面有I
    #                     begin = True
    #         elif 'I-' in _type:
    #             if t != p:  # B后面有I不等
    #                 begin = False
    #
    #             if (i+1 == n or 'I-' not in self._wd_vocab.index2tag(target[i+1])) and begin:
    #                 nb_correct += 1
    #
    #     return nb_correct, nb_gold, nb_pred

    def _calc_f1(self, nb_correct, nb_gold, nb_pred):
        precision = nb_correct / nb_pred
        recall = nb_correct / nb_gold
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _calc_acc(self, pred, target, non_pad_mask):
        '''
        :param pred:  (batch_size, seq_len)
        :param target:  (batch_size, seq_len)
        :param non_pad_mask: (batch_size, seq_len)
        :return:
        '''
        # 根据掩码选取有效元素
        pred = pred.masked_select(non_pad_mask)
        target = target.masked_select(non_pad_mask)
        assert pred.size() == target.size()
        # 判断实体是否匹配
        return self._exact_match(pred, target)

