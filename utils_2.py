"""
utils_2.py:
    数据处理中所需工具类和对应方法
    类Index/Vocabulary:实现元素与其对应索引的转换
    类GroupBatchRandomSampler:以组为单位，按batchsize取样并整合
"""
import string
import pickle
import torch
from torch.utils.data.sampler import *

#  实现元素与索引转换 
class Index(object):
    """
    index 与 key(word) 转换
    key2idx: key为索引,存储key在index中的位置
    idx2key: index为索引，存储key
    """
    # ======= 初始化idx和key,分别基于key和idx索引 =============
    def __init__(self):
        # key2idx: dict字典数据类型, 键值映射, 对应于实体中的每个词的起始idx
        self.key2idx = {}
        self.idx2key = []
    
    # ====== 根据不同类型索引获得key或idx ========
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.key2idx[key]
        if isinstance(key, int):
            return self.idx2key[key]
    
    # ==== 获得当前key的数目 ========
    def __len__(self):
        return len(self.idx2key)
    
    # ======= 添加新的key并更新其idx ========
    def add(self, key):
        if key not in self.key2idx:
            self.key2idx[key] = len(self.idx2key)
            self.idx2key.append(key)
        return self.key2idx[key]

    # ======== 保存对象中的key与idx至文件f中 ==========
    def save(self, f):
        with open(f, 'wt', encoding='utf-8') as fout:
            for index, key in enumerate(self.idx2key):
                fout.write(key + '\t' + str(index) + '\n')
    
    # ====== 逐一添加文件f(vocab_set.txt)中的词 ========
    def load_words(self, f):
        with open(f, 'rt', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                # 取出每行中的词
                key = line.split()[0]
                self.add(key)
    
    # ======== 逐一添加文件f(clause_set.txt)中的子句 ========
    def load_clauses(self, f):
        with open(f, 'rt', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                self.add(line)

# 继承Index, 构造单词集合
class Vocabulary(Index):
    """
    词表类型
    """
    # ========== 初始化,增加<pad>/<unk> ===========
    def __init__(self):
        super().__init__()
        self.add("<pad>")
        self.add("<unk>")

    # ========== 获得key在单词表中索引 ====================
    def __getitem__(self, key):
        if isinstance(key, str) and key not in self.key2idx:
            return self.key2idx["<unk>"]
        return super().__getitem__(key)

# 在每个组中按照batch_size取样并整合
class GroupBatchRandomSampler(object):
    """
    按batch随机取样
    """
    # ====== 初始化, 将每个组的数据(index)按batch_size划分并整合 ============
    def __init__(self, data_groups, batch_size, drop_last):
        self.batch_index = []
        # 按doc长度排列处理
        for data in data_groups:
            # SubsetRandomSampler:打乱排序
            # BatchSampler:以batch_size为单位添加
            # drop_last=False:保留后面少于一个batch的数据
            samples = BatchSampler(SubsetRandomSampler(data.indices), batch_size, drop_last=drop_last)
            self.batch_index.extend(list(samples))
    # ======= 打乱全部batch的顺序 ============
    def __iter__(self):
        length = len(self.batch_index)
        # torch.randperm(n):返回一个由0至n-1组成的随机排序的tensor
        return (self.batch_index[i] for i in torch.randperm(length))
    # ======= 获得batch总数 ===================
    def __len__(self):
        return len(self.batch_index)