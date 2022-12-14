import copy

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, split, com = False):
        super().__init__()

        if com == False:
            self.vocab = data['vocab']
        else:
            self.vocab = data['vocab_com']

        self.pad_index = len(self.vocab.keys()) if '<pad>' not in self.vocab.keys() else self.vocab['<pad>']
        print('padidx = '+ str(self.pad_index))
        self.max_len = data.get('max_len', 30)
        self.make_dataset(data, split, com)



    def make_dataset(self, data, split, com = False):
        # Data是包含了整个数据集的数据
        # 而我们只需要训练集/验证集/测试集的数据
        # 我们按照划分基准split里面的下标来确定加载哪部分的数据
        self.dataset = []
        if com == False:
            for idx in split:
                this_sentence_id = data['sentences_id'][idx]
                item = [
                    torch.LongTensor(self.pad_data(this_sentence_id[0])),
                    torch.LongTensor([data['labels'][idx]])
                ]
                self.dataset.append(item)
        else:
            for idx in split:
                this_sentence_id = data['sentences_id'][idx]
                for i in range(len(this_sentence_id[1])):

                    item = [
                        torch.LongTensor(self.pad_data(this_sentence_id[1][i])),
                        torch.LongTensor([data['labels'][idx]])
                    ]
                    self.dataset.append(item)

    def pad_data(self, seq):
        # 让序列长度最长只有max_len，不足就补pad，超过就截断
        if len(seq) < self.max_len:
            seq += [self.pad_index] * (self.max_len - len(seq))

        else:
            seq = seq[:self.max_len]
        return seq

    def get_pad_index(self):
        return self.pad_index

    def __getitem__(self, ix):
        # ix大于等于0，小于len(self.dataset)
        return self.dataset[ix]

    def __len__(self):
        # 一共有多少数据
        return len(self.dataset)



