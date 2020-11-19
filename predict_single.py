"""
predict_single.py:
    针对单条数据的预测

"""
import torch
from utils_2 import *
from gensim.models import Doc2Vec
import re
import jieba

PATH_MODEL = './model_2.pt'
# PATH_MODEL = './model_3.pt'
PATH_TAG_SET = './data/tag_set.txt'
PATH_EMOTION_LABELS = './data/emotion_labels.txt'
PATH_CLAUSE_SET = './data/clause_set.txt'
PATH_VOCAB_SET = './data/vocab_set.txt'
PATH_DOC_VEC_MODEL = './data/doc2vec'
# 每个子句对齐的最大词语个数
MAX_CLAUSE_LENGTH = 20

tag_set = Index()
emotion_labels = Index()
tag_set.load_words(PATH_TAG_SET)
emotion_labels.load_words(PATH_EMOTION_LABELS)

clause_set = Vocabulary()
clause_set.load_clauses(PATH_CLAUSE_SET)
vocab_set = Vocabulary()
vocab_set.load_words(PATH_VOCAB_SET)

# 中英文常用标点符号
end_flag_cn = ['，', '。', '？', '！', '……', '：', '‘', '’', '”', '“', '；', '——']
end_flag_eg = [',', '.', '?', '!', '...', ':', '\'', '\"', ';']

# ================ 划分子句 ========================================
def get_clause(doc):
    # 结束符号，包含中文和英文的
    end_flag = []
    end_flag.extend(end_flag_cn)
    end_flag.extend(end_flag_eg)

    doc_len = len(doc)
    sentence = []
    tmp_char = ''
    for idx, char in enumerate(doc):
        # 拼接字符
        tmp_char += char
        # 判断是否已经到了最后一位
        if (idx + 1) == doc_len:
            sentence.append(tmp_char)
            break
        # 判断此字符是否为结束符号
        if char in end_flag_cn or char in end_flag_eg:
            # 再判断下一个字符是否为结束符号，如果不是结束符号，则切分句子
            next_idx = idx + 1
            if not doc[next_idx] in end_flag_cn and not doc[next_idx] in end_flag_eg:
                sentence.append(tmp_char)
                tmp_char = ''
    return sentence

# ================= 根据index获取情绪-原因对 =============================================     
def get_tuples(tag_index):
    temp = {}
    tuples = []
    for idx, tag_id in enumerate(tag_index):
        # 0->tag_set["O"],不做处理
        if tag_id == 0:
            continue
        # 把二元组标签拆开
        emotion_label, role = tag_set[tag_id].split("-")
        # 该emotion未出现过
        if emotion_label not in temp:
            temp[emotion_label] = [[], []]
        temp[emotion_label][int(role) - 1].append(idx)
    for emotion_label in temp:
        # role1:情绪词(所在子句), role2:原因子句
        role1, role2 = temp[emotion_label]
        # 二者皆不为空
        if role1 and role2:
            # 从情绪子句出发，同其对应原因子句配对
            for e1 in role1:
                for e2 in role2:
                    tuples.append((e1, emotion_label, e2))
    return tuples

# =============== 获得相同词汇个数 =====================================
def get_num_similiar(sent1, sent2):
    num = 0
    for word1 in sent1:
        for word2 in sent2:
            if word1 == word2:
                num += 1
    return num

# =============== 获得sentence中所有word在to_idx中的对应index =========================
def get_words_idx(sentence, to_idx):
    return [to_idx[word] for word in sentence]

if __name__ == "__main__":
    with open(PATH_MODEL, 'rb') as f:
        model = torch.load(f)
    doc_input = "“当我看到建议被采纳，部委领导写给我的回信时，我知道我正在为这个国家的发展尽着一份力量。”27日，河北省邢台钢铁有限公司的普通工人\
白金跃拿着历年来国家各部委反馈给他的感谢信，激动地对中新网记者说，“27年来，国家公安部、国家工商总局、国家科学技术委员会、\
科技部、卫生部、国家发展改革委员会等部委均接受并采纳过的我的建议。”"
    doc_input1 = "得到公司的鼓励后，两人在同事面前可以大大方方牵手了，不仅如此，还会得到同事的祝福，他们觉得很开心。"
    doc_input2 = "生不如死是老吴患癌后常有的想法。由于大面积骨转移，老吴每天都在剧痛里挣扎。从2012年10月到现在，老吴已经和晚期肺癌搏斗了30个月，\
这个时间长度比医生最初给他的死亡判决已经超出了近10倍，深圳宁养院的医生王劲也感到惊诧。"
    doc_input3 = "小明生病住院，小红为此很难过。"

    # 得到划分后的子句
    clause_input = get_clause(doc_input)
    test_clause_set = []
    # 对子句预处理
    for i in range(len(clause_input)):
        # 保留汉字
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        clause_input[i] = re.sub(pattern, '', clause_input[i])
        # 去除空子句
        if clause_input[i] != '':
            test_clause_set.append(clause_input[i])
    # 分句结果
    # print(test_clause_set)

    # 对子句进行分词
    doc_length = len(test_clause_set)
    for i in range(doc_length):
        # 调用jieba分词
        word_list = list(jieba.cut(test_clause_set[i]))
        num_word = len(word_list)
        cur_word = ''
        temp = ''
        clause_i = 0
        for word_i in range(num_word):
            while cur_word != word_list[word_i]:
                cur_word += test_clause_set[i][clause_i]
                temp += test_clause_set[i][clause_i]
                clause_i += 1
            temp += ' '
            cur_word = ''
        test_clause_set[i] = temp.strip()

    # 保存文档索引序列
    doc_idx = []
    # 保存子句索引序列
    clause_idx = []
    # 遍历全部子句
    # 基于最大相似度获取预定义集合中的索引序列
    for test in test_clause_set:
        test = test.split(' ')
        # 加载doc2vec模型
        doc2vec_model = Doc2Vec.load(PATH_DOC_VEC_MODEL)
        # 得到测试数据的子句向量
        inferred_vector = doc2vec_model.infer_vector(doc_words=test,steps=100,alpha=0.025)
        # 找到预定义子句集合中最相近的10个
        sims = doc2vec_model.docvecs.most_similar([inferred_vector],topn=10)
        max_num = -1
        max_idx = 0
        for idx,sim in sims:
            # idx+2:跳过clause前面的<pad>和<unk>
            num = get_num_similiar(clause_set[int(idx+2)].split(' '),test)
            if num > max_num:
                max_idx = int(idx)
                max_num = num
        doc_idx.append(max_idx)
        # 获得每个词在预定义单词集合中的idx
        words_idx = get_words_idx(test, vocab_set)
        # 对于大多数长度不超过MAX_CLAUSE_LENGTH(=20)的子句,用vocab_set["<pad>"](=0)填充
        if len(words_idx) < MAX_CLAUSE_LENGTH:
            words_idx.extend([vocab_set["<pad>"]]*(MAX_CLAUSE_LENGTH-len(words_idx)))
        else:
            # 对于个别长度大于MAX_CLAUSE_LENGTH(=20)的子句,取前10个和后10个单词
            words_idx = words_idx[0:10] + words_idx[-10:]
        clause_idx.append(words_idx)

# =========== 调用训练好的模型进行预测 =================================
    device = torch.device("cuda")
    # model.eval():不启用 BatchNormalization 和 Dropout
    model.eval()
    doc = torch.LongTensor([doc_idx]).to(device)
    clause = torch.LongTensor([clause_idx]).to(device)
    output = model(doc, clause)
    output = torch.argmax(output, dim=-1)
    # 获得output真实长度，即舍去<pad>部分
    # np.tolist():将数组矩阵转化为列表
    out = output[0].tolist()
    # 得到三元组形式
    out_tuples = get_tuples(out)
    # print(out_tuples)
    # 将索引转为测试数据中的子句
    res = []
    for out_tuple in out_tuples:
        e1 = test_clause_set[out_tuple[0]]
        e2 = test_clause_set[out_tuple[2]]
        emotion = out_tuple[1]
        res.append([e1,emotion,e2])
    print('\ninput:\n ',doc_input)
    print('output:')
    for pair in res:
        print(pair)