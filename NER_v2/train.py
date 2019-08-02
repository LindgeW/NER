import torch
import torch.nn as nn
import numpy as np
from config.Config import data_path_conf, arg_conf
from datautil.dataloader import load_data
from vocab.Vocab import create_vocabs
from modules.bilstm_crf import BiLSTMCRF
from tagger import SequenceTagger


if __name__ == '__main__':
    np.random.seed(3347)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    # 加载数据
    data_opts = data_path_conf('config/data_path.json')
    train_data = load_data(data_opts['data']['train_path'])
    dev_data = load_data(data_opts['data']['dev_path'])
    test_data = load_data(data_opts['data']['test_path'])
    print('train size:%d, dev size:%d, test size:%d' % (len(train_data), len(dev_data), len(test_data)))
    # 参数配置
    args = arg_conf()
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')

    print('GPU available:', torch.cuda.is_available())
    print('CuDNN available', torch.backends.cudnn.enabled)
    print('GPU number: ', torch.cuda.device_count())

    # 构建词表
    wd_vocab, ch_vocab = create_vocabs(data_opts['data']['train_path'])
    wd_embedding_weights = wd_vocab.get_embedding_weights(data_opts['data']['embedding_weights'])
    signal_embedding_weights = wd_vocab.build_signal_embed()

    # 创建模型
    args.pad = wd_vocab.PAD
    args.tag_size = wd_vocab.tag_size
    args.char_vocab_size = ch_vocab.vocab_size
    bilstm_crf = BiLSTMCRF(args, wd_embedding_weights, signal_embedding_weights).to(args.device)
    # GPU并行
    # if torch.cuda.device_count() > 1:
    #     bilstm_crf = nn.DataParallel(bilstm_crf, device_ids=[0, 1])  # 在指定编号的GPU上运行

    tagger = SequenceTagger(bilstm_crf, args, wd_vocab, ch_vocab)
    tagger.summary()

    # 训练
    tagger.train(train_data, dev_data)

    # 评估
    tagger.evaluate(test_data)
