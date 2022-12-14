import copy
import torch.optim
from torch.utils.data import DataLoader
import extraction
import dataset
import gensim
import numpy as np
class AverageMeter(object):
    """
    用于储存与计算平均值
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, multiply=True):
        self.val = val
        if multiply:
            self.sum += val * n
        else:
            self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def training(model, loader, crit, optim, device):
    # 模型调成训练模式
    model.train()
    # 把模型移到指定设备
    model.to(device)
    # 用于记录损失和正确率
    meter_loss, meter_acc = AverageMeter(), AverageMeter()

    for data in loader:
        # 清空梯度
        optim.zero_grad()
        # 获取数据并将其移至指定设备中, cpu / gpu
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.view(-1)
        # 将输入送入网络，获得输出
        outputs = model(inputs)
        # 计算损失
        loss = crit(outputs, labels)
        # 反向传播，计算梯度
        loss.backward()
        # 更新网络参数
        optim.step()

        # 记录损失
        num_sample = inputs.size(0)
        meter_loss.update(loss.item(), num_sample)
        # 记录预测正确率
        preds = outputs.max(dim=1)[1]  # 网络预测的类别结果
        correct = (preds == labels).sum()  # 计算预测的正确个数
        meter_acc.update(correct.item(), num_sample, multiply=False)

    # 返回训练集的平均损失和平均正确率
    return meter_loss.avg, meter_acc.avg


@torch.no_grad()
def evaluate(model, loader, crit, device):
    # 模型调成评估模式
    model.eval()
    # 把模型移到指定设备
    model.to(device)
    # 用于记录损失和正确率
    meter_loss, meter_acc = AverageMeter(), AverageMeter()
    out_pre = []
    for data in loader:
        # 获取数据并将其移至指定设备中, cpu / gpu
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.view(-1)
        # 将输入送入网络，获得输出
        outputs = model(inputs)
        out_pre.append(outputs.cpu().numpy())

        # 计算并记录损失
        loss = crit(outputs, labels)
        num_sample = inputs.size(0)
        meter_loss.update(loss.item(), num_sample)
        # 记录预测正确率
        preds = outputs.max(dim=1)[1]  # 网络预测的类别结果
        correct = (preds == labels).sum()  # 计算预测的正确个数
        meter_acc.update(correct.item(), num_sample, multiply=False)

    return meter_loss.avg, meter_acc.avg, out_pre



def evaluate_final(ucom_pre, com_pre, split, test_loader):
    label_f = []
    cnt_f = []
    com_f = []
    ucom_f = []

    for t in split['test']:
        cnt_f.append(len(data['sentences_id'][t][1]))

    for t in test_loader:
        _, tabels = t
        tabels = tabels.view(-1)

        label_f += list(tabels.numpy())

    for t in com_pre:
        com_f += list(t)

    for t in ucom_pre:
        ucom_f += list(t)

    tot = 0
    acc = 0
    j = 0

    c020 = 0
    c021 = 0
    c120 = 0
    c121 = 0

    for i in range(len(split['test'])):
        res = copy.deepcopy(ucom_f[i])

        for _ in range(cnt_f[i]):
            res += com_f[j]
            res += ucom_f[i]
            j += 1

        if (res[0] > res[1]):
            ans = 0
        else:
            ans = 1

        if ans == label_f[i]:
            acc += 1
            if ans == 0:
                c020 += 1
            else:
                c121 += 1
        else:
            if ans == 0:
                c120 += 1
            else:
                c021 += 1
        tot += 1

    print('res!')
    print(c020)
    print(c021)
    print(c120)
    print(c121)
    print(acc)
    print(tot)
    return acc / tot


def get_loader(data, split, batch_size=64, class_func=dataset.MyDataset, com=False):
    # split.keys() 包括 'train', 'vali', 'test'
    # 所以此函数是为了拿到训练集，验证集和测试集的数据加载器
    loader = []
    for mode in split.keys():
        print('splitmod' + str(mode))
        # split[mode]指定了要取data的哪些数据
        dataset = class_func(data, split[mode], com)
        # Dataloader可帮助我们一次性取batch_size个样本出来
        loader.append(
            DataLoader(dataset,
                       batch_size=batch_size,
                       shuffle=True if mode == 'train' else False)
        )
    return loader


data, split, vocab, vocab_com = extraction.main()

if '<pad>' not in vocab.keys():
    vocab['<pad>'] = len(vocab.keys())

if '<cls>' not in vocab.keys():
    vocab['<cls>'] = len(vocab.keys())

if '<pad>' not in vocab_com.keys():
    vocab_com['<pad>'] = len(vocab_com.keys())

if '<cls>' not in vocab_com.keys():
    vocab_com['<cls>'] = len(vocab_com.keys())



# 参数
num_epochs = 10
learning_rate = 0.005
learning_rate_com = 0.002
batch_size = 128
hidden_size = 50
num_heads = 10
num_layers = 1
attn_dropout = 0.15
dropout = 0.4
attn_dropout_com = 0.15
dropout_com = 0.4
num_epochs_com = 5
data['max_len'] = 100

# 运行的设备
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 数据加载器
train_loader_com, vali_loader_com, test_loader_com = get_loader(data, split, batch_size=batch_size,
                                                                class_func=dataset.MyDataset, com=True)
train_loader, vali_loader, test_loader = get_loader(data, split, batch_size=batch_size, class_func=dataset.MyDataset,
                                                    com=False)
def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs

word2vec = build_word2vec('./wiki_word2vec_50.bin', vocab)
word2vec_com = build_word2vec('./wiki_word2vec_50.bin', vocab_com)

# 模型实例化
import model

model_ucom = model.TFModel(
    word2vec,
    len(vocab),
    hidden_size,
    data['max_len'],
    pad_index=train_loader.dataset.get_pad_index(),
    cls_index=vocab['<cls>'],
    num_heads=num_heads,
    num_layers=num_layers,
    attn_dropout=attn_dropout,
    dropout=dropout
)

model_com = model.TFModel(
    word2vec_com,
    len(vocab_com),
    hidden_size,
    data['max_len'],
    pad_index=train_loader_com.dataset.get_pad_index(),
    cls_index=vocab_com['<cls>'],
    num_heads=num_heads,
    num_layers=num_layers,
    attn_dropout=attn_dropout_com,
    dropout=dropout_com
)


# 损失函数 -- 交叉熵
crit = torch.nn.NLLLoss()
# 优化方法
optimizer = torch.optim.Adam(model_ucom.parameters(), lr=learning_rate)
optimizer_com = torch.optim.Adam(model_com.parameters(), lr=learning_rate_com)
records = []



for epoch in range(num_epochs):
    # 训练
    train_loss, train_acc= training(model_ucom, train_loader, crit, optimizer, device)
    # 验证
    vali_loss, vali_acc, _ = evaluate(model_ucom, vali_loader, crit, device)

    
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate * (0.1 ** (epoch // 5))
    
    # 打印消息
    print('第{}轮，训练集损失：{:.2f}, 训练集准确率：{:.2f}, 验证集损失：{:.2f}, 验证集准确率: {:.2f}'.format(
        epoch, train_loss, train_acc, vali_loss, vali_acc))
    # 储存信息以便可视化
    records.append([train_loss, train_acc, vali_loss, vali_acc])
torch.save(model_ucom.state_dict(), 'state')



for epoch in range(num_epochs_com):
    # 训练
    train_loss, train_acc = training(model_com, train_loader_com, crit, optimizer_com, device)
    # 验证
    vali_loss, vali_acc, _ = evaluate(model_com, vali_loader_com, crit, device)
    # 打印消息
    
    for param_group in optimizer_com.param_groups:
        param_group['lr'] = learning_rate_com * (0.1 ** (epoch // 3))
    
    print('第{}轮，训练集损失：{:.2f}, 训练集准确率：{:.2f}, 验证集损失：{:.2f}, 验证集准确率: {:.2f}'.format(
        epoch, train_loss, train_acc, vali_loss, vali_acc))
    # 储存信息以便可视化
    records.append([train_loss, train_acc, vali_loss, vali_acc])

torch.save(model_com.state_dict(), 'state_com')



# model_ucom.load_state_dict(torch.load('state'))
# model_com.load_state_dict(torch.load('state_com'))
# 测试
_, test_acc1, ucom_pre = evaluate(model_ucom, test_loader, crit, device)
_, test_acc2, com_pre = evaluate(model_com, test_loader_com, crit, device)



evaluate_final(ucom_pre, com_pre, split, test_loader)


