"""
doc2vec_2.py:
    1.获取整合后的子句集合clause_set.txt
    2.借助gensim包训练Doc2Vec模型获得子句向量:
        clause_vec.txt--------子句向量
        clause_set.txt--------语料库子句表
"""
import gensim
from gensim.models.doc2vec import LabeledSentence, TaggedDocument
from gensim.models import Doc2Vec
import os

PATH_DATA_SOURCE = './data/clause_keywords_emotion.txt'
PATH_CLAUSE_SET = './data/clause_set.txt'
PATH_DOC_VEC_MODEL = './data/doc2vec'
PATH_DOC_VEC_TXT = './data/clause_vec.txt'

# ==================== 获得doc2vec训练文本(除去重复,保留存在包含关系的子句) =====================================================
def get_clause_set(fin, fout):
    # 保存带有包含关系且不重复的子句
    clause_set = []
    # 保存doc2vec训练语料
    doc_train = []
    # 每行子句对应的一个id
    index = 0
    for line in fin:
        # 移除每条数据中开头或结尾的空格或换行符
        line = line.strip().split(',')
        if not line:
            continue
        # 获得该行中的子句
        clause = line[-1].strip()
        if clause not in clause_set:
            clause_set.append(clause)
    for clause in clause_set:
        doc_train.append(TaggedDocument(clause.split(' '), tags=[index]))
        # 将子句逐一写入fout中
        fout.write(clause + '\n')
        index += 1
    return doc_train

if __name__ == '__main__':
    # 获取预处理的语料
    if not os.path.exists(PATH_CLAUSE_SET):
        with open(PATH_CLAUSE_SET, 'wt', encoding='utf-8') as fout:
            with open(PATH_DATA_SOURCE, 'rt', encoding='utf-8') as fin:
                doc_tarin = get_clause_set(fin, fout)
        # print(len(doc_tarin))# 27007
    # sample:高频词被随机降低采样的阈值
    model= gensim.models.Doc2Vec(doc_tarin, vector_size=300, min_count=1, windows=3)
    model.train(doc_tarin,total_examples=model.corpus_count,epochs=10)
    model.save(PATH_DOC_VEC_MODEL)
    
    model = Doc2Vec.load(PATH_DOC_VEC_MODEL)
    model.save_word2vec_format(fname=PATH_DOC_VEC_TXT, doctag_vec=True, word_vec=False)
    # print(len(model.docvecs.vectors_docs))# 27007