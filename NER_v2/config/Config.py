import json
import argparse


def data_path_conf(path):
    with open(path, 'r', encoding='utf-8') as fin:
        opts = json.load(fin)
    return opts


def arg_conf():
    parser = argparse.ArgumentParser("Text Classification")
    # 通用参数
    parser.add_argument("--cuda", type=int, default=-1, help="which device, default cpu")
    parser.add_argument("--patience", type=int, default=5, help='early-stopping patient iters')
    # 数据参数
    parser.add_argument("--epoch", type=int, default=20, help="Iter number of all data")
    parser.add_argument("--batch_size", type=int, default=32, help="batch data size")

    # 优化器参数参数
    parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-7, help="weight decay when update")

    # 模型参数
    parser.add_argument("--filter_size", type=int, default=50, help="cnn filter size")
    parser.add_argument("--hidden_size", type=int, default=100, help="rnn hidden size")
    parser.add_argument("--num_layers", type=int, default=1, help="the number of rnn layer")

    parser.add_argument("--dropout_embed", type=float, default=0.5, help="drop out of embedding layer")
    parser.add_argument("--dropout_linear", type=float, default=0.5, help="drop out of linear layer")

    # 字符参数配置
    parser.add_argument("--char_feature_size", type=int, default=50, help="char embedding dim")
    parser.add_argument("--char_embed_dim", type=int, default=400, help="char hidden_size")

    args = parser.parse_args()

    # 打印出对象的属性和方法
    print(vars(args))
    return args
