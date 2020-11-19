"""
word2vec_2.py:
    1.获取整合后的文档集合doc_set.txt
    2.借助gensim包训练Word2Vec模型获得词向量:
        word2vec.txt ------------ 词向量(24094*200)
        vocab_set.txt ----------- 语料库词汇表
"""
from gensim.models.word2vec import LineSentence, Word2Vec
import os

PATH_DATA_SOURCE = './data/clause_keywords_emotion.txt'
PATH_DOC_SET = './data/doc_set.txt'
PATH_WORD_VEC_MODEL = './data/word2vec'
PATH_WORD_VEC_TXT = './data/word_vec.txt'
PATH_VOCAB_SET = './data/vocab_set.txt'

# ======================= 将fin中内容非重复的文档逐一写入fout中 =======================================================
def get_doc_set(fin, fout):
    # doc集合
    doc_set = []
    # 保存文档doc总数
    doc_num = 0
    # 保存数据集中子句最大长度，即单个子句最大单词数目
    max_clause_len = 0
    # 保存前一子句的index
    idx_clause_pre = 0
    # 保存当前doc的内容
    doc = ''
    # 保存数据集中doc最大长度，即所有子句中最大的index
    max_doc_len = 0
    # 保存不重复doc中的子句总数
    num_lines = 0
    # 当前doc中的子句总数
    cur_lines = 0
    for line in fin:
        # 移除每条数据中开头或结尾的空格和换行符
        line = line.strip().split(',')
        if not line:
            continue
        # 求单个doc中子句最大数目
        if int(line[2]) > max_doc_len:
            max_doc_len = int(line[2])
        # line[0]:该段doc的index
        # 与前一子句在同一doc内
        if line[0] == idx_clause_pre:
            cur_lines += 1
            doc = doc + ' ' + line[-1].strip()
            # 求单个子句的最大长度
            if len(line[-1].strip()) > max_clause_len:
                max_clause_len = len(line[-1].strip())
        else:
            # 与已保存的doc内容不同, 则存入doc_set内
            if doc not in doc_set and idx_clause_pre != 0:
                doc_set.append(doc)
                doc_num += 1
                num_lines += cur_lines
            # 更新当前状态信息
            cur_lines = 1
            doc = line[-1].strip()
            idx_clause_pre = line[0]
    # 写入最后一条doc
    if doc not in doc_set:
        doc_set.append(doc)
        doc_num += 1
        num_lines += cur_lines
    # 将doc逐一写入fout中
    for doc in doc_set:
        fout.write(doc + '\n')
    # # 单个子句最大长度(单词个数)
    # print(max_clause_len) # 117
    # # 单个文本最大长度(子句个数)
    # print(max_doc_len) # 73
    # # 子句总数
    # print(num_lines) # 28553
    return doc_num

if __name__ == "__main__":
    # 整合训练数据中的文本
    if not os.path.exists(PATH_DOC_SET):
        with open(PATH_DOC_SET, 'wt', encoding='utf-8') as fout:
            with open(PATH_DATA_SOURCE, 'rt', encoding='utf-8') as fin:
                # 返回文档总数
                doc_num = get_doc_set(fin, fout)
                # print(doc_num) # 1933
    # 加载语料
    sentences = LineSentence(PATH_DOC_SET)
    model = Word2Vec(sentences, size=200, sg=1, min_count=1)
    model.save(PATH_WORD_VEC_MODEL)
    # word2vec.txt:词向量表(24094*200)
    # vocab.txt:保存语料库中所有词
    model.wv.save_word2vec_format(PATH_WORD_VEC_TXT, fvocab=PATH_VOCAB_SET)