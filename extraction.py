import copy
import json, os
from tqdm import tqdm
import re  # 正则表达式的包
import jieba  # 结巴分词包
import numpy as np
from collections import Counter  # 搜集器，可以让统计词频更简单
import gensim

# 将文本中的标点符号过滤掉
def filter_punc(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）：:；“”】》《-【\][]", "", sentence.strip())
    return sentence

import matplotlib.pyplot as plt

# 扫描所有的文本，分词、建立词典，分出是谣言还是非谣言，is_filter可以过滤是否筛选掉标点符号
def Prepare_data(good_file, bad_file, is_filter=True, threshold=3, stop_words = []):
    all_words = []  # 存储所有的单词
    all_words_com = [] # 存储所有评论单词
    pos_sentences = []  # 存储非谣言
    neg_sentences = []  # 存储谣言
    print(stop_words)
    for line in good_file:
        if is_filter:
            # 过滤标点符号
            line[0] = filter_punc(line[0])
            for i in range(len(line[1])):
                line[1][i] = filter_punc(line[1][i])
        words = jieba.lcut(line[0])
        for t in words:
            if t in stop_words:
                words.remove(t)
        if len(words) > 0:
            all_words += words
            all_com = []
            for i in range(len(line[1])):
                words_com = jieba.lcut(line[1][i])
                for t in words_com:
                    if t in stop_words:
                        words_com.remove(t)
                if len(words_com) > 2:
                    all_words_com += words_com
                    all_com.append(words_com)

            pos_sentences.append([words,all_com])


    for line in bad_file:
        if is_filter:
            # 过滤标点符号
            line[0] = filter_punc(line[0])
            for i in range(len(line[1])):
                line[1][i] = filter_punc(line[1][i])
        words = jieba.lcut(line[0])
        for t in words:
            if t in stop_words:
                words.remove(t)
        if len(words) > 0:
            all_words += words
            all_com = []
            for i in range(len(line[1])):
                words_com = jieba.lcut(line[1][i])
                for t in words_com:
                    if t in stop_words:
                        words_com.remove(t)
                if len(words_com) > 2:
                    all_words_com += words_com
                    all_com.append(words_com)

            neg_sentences.append([words,all_com])

    # 建立词典，只保留频次大于threshold的单词
    vocab = {'<unk>': 0}
    cnt = Counter(all_words)
    for word, freq in cnt.items():
        if freq > threshold:
            vocab[word] = len(vocab)

    vocab_com = {'<unk>': 0}
    cnt = Counter(all_words_com)
    for word, freq in cnt.items():
        if freq > threshold:
            vocab_com[word] = len(vocab_com)

    print('过滤掉词频 <= {}的单词后，字典大小：{}，评论字典大小'.format(threshold, len(vocab), len(vocab_com)))
    return pos_sentences, neg_sentences, vocab, vocab_com




def sentence2vocab(sentence, vocab):
    new_sentence = []
    for word in sentence:
        new_sentence.append(vocab[word] if word in vocab else vocab['<unk>'])
    return new_sentence

def main():
    # 数据来源文件夹 -- 内含多个json文件
    non_rumor = './Chinese_Rumor_Dataset/CED_Dataset/non-rumor-repost'
    rumor = './Chinese_Rumor_Dataset/CED_Dataset/rumor-repost'
    original = './Chinese_Rumor_Dataset/CED_Dataset/original-microblog'


    non_rumor_data = []
    rumor_data = []

    with open('stop.txt', encoding='utf-8') as file:
        stopw = file.read().split('\n')

    # 遍历文件夹，读取文本数据
    print('开始读取数据')

    for file in tqdm(os.listdir(original)):
        try:
            data = json.load(open(os.path.join(original, file), 'rb'))['text']
        except:
            continue

        non_rumor_data_com = []
        rumor_data_com = []
        is_rumor = (file in os.listdir(rumor))
        if is_rumor:
            com_json = json.load(open(os.path.join(rumor, file), 'rb'))
            for it in com_json:
                rumor_data_com.append(it['text'])
            data = [data, rumor_data_com]

            rumor_data.append(data)

        else:

            com_json = json.load(open(os.path.join(non_rumor, file), 'rb'))
            for it in com_json:
                non_rumor_data_com.append(it['text'])
            data = [data, non_rumor_data_com]

            non_rumor_data.append(data)

    print('结束, 有{}条谣言, 有{}条非谣言'.format(len(rumor_data), len(non_rumor_data)))

    labels = []  # 标签
    sentences = []  # 原始句子，调试用
    sentences_id = []  # 原始句子对应的index列表


    pos_sentences, neg_sentences, vocab ,vocab_com = Prepare_data(non_rumor_data, rumor_data, True, threshold=3, stop_words=stopw)
    flag = False
    # 处理非谣言
    for sentence in pos_sentences:
        sentences.append(copy.deepcopy(sentence))

        sentence[0] = sentence2vocab(sentence[0],vocab)
        #sentence[0].insert(0,len(vocab) + min(len(sentence[1]) / 5,100))
        if flag == False:
            print('sentence!')
            print(sentence[0])
            flag = True
        for i in range(len(sentence[1])):
            sentence[1][i] = sentence2vocab(sentence[1][i],vocab_com)
            #sentence[1][i].insert(0,len(vocab_com) + min(len(sentence[1]) / 5,100))
        sentences_id.append(sentence)
        labels.append(0)  # 正标签为0


    # 处理谣言
    for sentence in neg_sentences:
        sentences.append(sentence)

        sentence[0] = sentence2vocab(sentence[0], vocab)
        #sentence[0].insert(0,len(vocab) + min(len(sentence[1]) / 5,100))
        for i in range(len(sentence[1])):
            sentence[1][i] = sentence2vocab(sentence[1][i], vocab_com)
            #sentence[1][i].insert(0,len(vocab_com) + min(len(sentence[1]) / 5,100))
        sentences_id.append(sentence)
        labels.append(1)  # 正标签为0



    # 打乱所有的数据顺序，形成数据集
    # indices为所有数据下标的一个全排列
    np.random.seed(0)
    indices = np.random.permutation(len(sentences_id))
    #避免数据泄露
    # 对整个数据集进行划分，分为：训练集、验证集和测试集，这里是2:1:1
    test_size = len(sentences_id) // 4

    data = {
        #'bow': bow,  # 词袋数据
        'labels': labels,  # 标签
        'sentences_id': sentences_id,  # 句子对应的下标列表
        'sentences': sentences,  # 句子
        'vocab': vocab,  # 词典,
        'vocab_com': vocab_com # 评论词典
    }
    split = {
        'train': indices[2 * test_size:],
        'vali': indices[:test_size],
        'test': indices[test_size:2 * test_size]
    }

    k_test = 0

    # 测试一下划分情况
    for key, indices in split.items():
        count = [0, 0]
        for idx in indices:
            count[labels[idx]] += 1
            idx = (idx + k_test) % len(sentences_id)
        print(key, '非谣言有{}条，谣言有{}条'.format(count[0], count[1]))

    return data, split, vocab, vocab_com

