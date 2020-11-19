"""
tagging_2.py: 
    1.对文本做标注
        使用"情绪-1/情绪-2"做子句级别的标签
        情绪类比关系，表示由1子句导致2子句的某种情感
        role1、role2表示对应的两个实体(1---情绪词(所在子句)、2---原因子句)
    2.获得模型的训练输入三元组:(doc_idx, clause_idx, tag_idx)
    doc_idx: num_clause*1
            代表文档索引
            单个数据对应clause_set.txt
    clause_idx: num_clause*num_word
            代表子句索引
            单个数据对应vocab_set.txt
    tag_idx: num_clause*1
            代表标签索引
            单个数据对应tag_set.txt
"""
from utils_2 import *
import math
# 每个文档对应的最大子句数
MAX_DOC_LENGTH = 40
# 每个子句对齐的最大词语个数
MAX_CLAUSE_LENGTH = 20

PATH_DATA_SOURCE = './data/clause_keywords_emotion.txt'
PATH_CLAUSE_SET = './data/clause_set.txt'
PATH_VOCAB_SET = './data/vocab_set.txt'
PATH_TAG_SET = './data/tag_set.txt'
PATH_EMOTION_LABELS = './data/emotion_labels.txt'
PATH_DATA_SET = './data/data_set.pk'
PATH_DATA_NEW_SET = './data/data_new_set.pk'
# =========== 获得sentence中所有word在to_idx中的对应index ====================================
def get_words_idx(sentence, to_idx):
    sentence = sentence.split(' ')
    return [to_idx[word] for word in sentence]

# ========= 针对某一emotion，向tag_set中其对应的二元组标签: 情绪+角色(e.g. happiness-1) ===============
def get_tag_set(tag_set, emotion_labels):
    for emotion in emotion_labels:
        for role in "12":
            tag_set.add("-".join([emotion, role]))
    return len(tag_set)

# =========== 更新标签序列,index小的标签覆盖大标签==================================================
def get_new_list(list1, list2):
    overlap = False
    if len(list1) != len(list2):
        print('Error: Somthing Wrong in the length of tag_idx')
        return -1
    new_list = []
    length = len(list1)
    for i in range(length):
        if list1[i]*list2[i] != 0:
            elemt = min(list1[i], list2[i])
            overlap = True
        else:
            elemt = max(list1[i], list2[i])
        new_list.append(elemt)
    return new_list, overlap

# ============== 预处理数据集 ==============================================
def prepare_data_set(fin, clause_set, vocab_set, tag_set, emotion_labels):
    # 出现标签覆盖的文档总数
    #   1. 原因标签被情绪标签覆盖
    #   2. 被编号靠前的标签覆盖
    num_overlap = 0
    dataset = []
    # 第一次遍历获得所有出现的情绪类别
    for line in fin:
        line = line.strip().split(',')
        if not line:
            continue        
        emotion = line[1]
        if emotion not in emotion_labels:
            emotion_labels.add(emotion)
    # 数据集中原始文档总数
    num_doc = line[0]
    # print(num_doc)# 2105
    # 获得所有可能的二元组标签
    num_tag = get_tag_set(tag_set, emotion_labels)
    # print(num_tag) # 13
    # print(len(emotion_labels)) # 6

    # 指针返回文件首
    fin.seek(0)
    # 当前文档索引: 长度num_clause(MAX_DOC_LENGTH = 40)
    doc_idx = []
    # 前一个文档的索引
    doc_pre_idx = []
    # 子句索引: 长度num_word(MAX_CLAUSE_LENGTH = 20)
    clause_idx = []
    # 标签索引
    tag_idx = []
    # 前一个文档标签的索引，处理多标签问题
    # (数据集中若同一个文档包含多个情感，文档会重复出现，分多次标注)
    tag_pre_idx = []

    # 当前子句的idx
    idx_cur = 1
    # 前一个子句的idx
    idx_pre = 0
    # 当前doc标签覆盖状态
    overlap = False
    # 前一个doc标签覆盖状态
    overlap_pre = False
    # 重复文本数目(针对相邻数据)
    repeat = 0
    # 超过MAX_CLAUSE_LENGTH的子句数
    clause_over = 0
    # 第二次遍历获取训练三元组数据
    for line2 in fin:
        # 移除每条数据中开头或结尾的空格或换行符
        line2 = line2.strip().split(',')
        if not line2:
            continue
        # 子句文本
        clause = line2[-1].strip()
        # 该段文档标注的情绪
        emotion = line2[1]
        # 该段文档的情绪子句标注
        # (= 0表示情绪词位于该子句中，其他值表示距该子句的距离)
        tag_emotion = int(line2[4])
        # 该段文档的原因子句标注
        # (= yes/no)
        tag_reason = line2[5]
        # 当前子句的idx
        idx_cur = int(line2[0])

        # 获得子句级标签
        tag = 'O'
        if tag_emotion == 0:
            role = '1'
            tag = '-'.join([emotion, role])
        if tag_reason == 'yes':
            if tag != 'O':
                # role2(reason)被role1(emotion)覆盖
                overlap = True
            else:
                role = '2'
                tag = '-'.join([emotion, role])
        
        # 与前一个子句不为同一doc
        # 读入dataset，更新前一个doc_idx、clause_idx、tag_idx
        if idx_cur != idx_pre:
            # 若跟前一个doc的文本相同，更新原dataset最后一条数据
            # 不对doc_idx和clause_idx做处理
            # tag_idx按“无变有，大变小”的准则覆盖合并
            if doc_pre_idx == doc_idx and dataset != []:
                dataset = dataset[0:-1]
                repeat += 1
                # 文档overlap重复计算
                if overlap_pre:
                    num_overlap -= 1
                tag_idx, overlap2 = get_new_list(tag_idx, tag_pre_idx)
                overlap = overlap or overlap2
            if overlap:
                num_overlap += 1
            if (doc_idx, clause_idx, tag_idx) in dataset:
                repeat += 1
            if idx_pre!=0 and (doc_idx, clause_idx, tag_idx) not in dataset:
                dataset.append((doc_idx, clause_idx, tag_idx))

            # 更新当前doc相关参数
            doc_pre_idx = doc_idx
            tag_pre_idx = tag_idx
            doc_idx = []
            tag_idx = []
            doc_idx.append(clause_set[clause])
            tag_idx.append(tag_set[tag])

            idx_pre = idx_cur
            overlap_pre = overlap
            overlap = False
            
            # 重置当前clause
            clause_idx = []
            words_idx = get_words_idx(clause, vocab_set)
            # 基于每个单词在vocab_set中的索引,对于大多数长度不超过MAX_CLAUSE_LENGTH(=20)的子句,不足则用vocab_set["<pad>"](=0)填充
            if len(words_idx) < MAX_CLAUSE_LENGTH:
                words_idx.extend([vocab_set["<pad>"]]*(MAX_CLAUSE_LENGTH-len(words_idx)))
            else:
                # 对于个别长度大于MAX_CLAUSE_LENGTH(=20)的子句,取前10个和后10个单词
                clause_over += 1
                words_idx = words_idx[0:10] + words_idx[-10:]
            clause_idx.append(words_idx)
        
        # 当前子句还在同一个doc中
        else:
            # 超过设定的长度，优先保留中间部分的子句
            if len(doc_idx) >= MAX_DOC_LENGTH:
                doc_idx = doc_idx[1:-1]
                tag_idx = tag_idx[1:-1]
                clause_idx = clause_idx[1:-1]
            doc_idx.append(clause_set[clause])
            tag_idx.append(tag_set[tag])
            words_idx = get_words_idx(clause, vocab_set)

            # 基于每个单词在vocab_set中的索引,对于大多数长度不超过MAX_CLAUSE_LENGTH(=20)的子句,不足则用vocab_set["<pad>"](=0)填充
            if len(words_idx) < MAX_CLAUSE_LENGTH:
                words_idx.extend([vocab_set["<pad>"]]*(MAX_CLAUSE_LENGTH-len(words_idx)))
            else:
                clause_over += 1
                # 对于个别长度大于MAX_CLAUSE_LENGTH(=20)的子句,取前10个和后10个单词
                words_idx = words_idx[0:10] + words_idx[-10:]
            clause_idx.append(words_idx)
    # 处理最后一个doc
    # 跟前一个doc的文本相同
    if doc_pre_idx == doc_idx:
        dataset = dataset[0:-1]
        repeat += 1
        tag_idx, overlap2 = get_new_list(tag_idx, tag_pre_idx)
        overlap = overlap or overlap2
    if (doc_idx, clause_idx, tag_idx) in dataset:
            repeat += 1
    if (doc_idx, clause_idx, tag_idx) not in dataset:
        dataset.append((doc_idx, clause_idx, tag_idx))
    # print('clause_over:',clause_over) # 105
    # print('repeat:',repeat) # 160
    return dataset, num_overlap

# ============ 使用pickle.dump(),将对象保存为二进制数据 =====================
def data_save(obj, path):
    # 其中file必须以二进制可写模式打开,即“wb”
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

# ============= 使用pickle.load(),解析文件中的二进制数据 ====================
def data_load(path): 
    # 其中file必须以二进制可读模式打开,即“rb”
    with open(path, 'rb') as f:
        return pickle.load(f)

# ============ 统计不同情绪类别的子句数目 ====================================
def get_num_relaions(data, tag_set, num_emotion):
    for line in data:
        # 得到每一个子句的标签索引
        for clause_tag_idx in line[2]:
            tag = tag_set[clause_tag_idx]
            emotion = tag.split('-')[0]
            num_emotion[emotion] += 1

# ============== 得到不同情绪类别的训练数据 =================================
def get_data_emotions(data,data_emotion):
    for line in data:
        # 得到每一个子句的标签索引
        for clause_tag_idx in line[2]:
            tag = tag_set[clause_tag_idx]
            emotion = tag.split('-')[0]
            if emotion == 'happiness':
                data_emotion['happiness'].append(line)
                break
            elif emotion == 'sadness':
                data_emotion['sadness'].append(line)
                break
            elif emotion == 'anger':
                data_emotion['anger'].append(line)
                break
            elif emotion == 'disgust':
                data_emotion['disgust'].append(line)
                break
            elif emotion == 'fear':
                data_emotion['fear'].append(line)
                break
            elif emotion == 'surprise':
                data_emotion['surprise'].append(line)
                break

# =============== 得到各个情绪均匀分布的数据集 ======================================
def get_new_dataset(data_new_set, interval_emotion, data_emotion, total_length):
    index = 1
    while index <= total_length:
        for emo in interval_emotion:
            if index % interval_emotion[emo] == 0 and data_emotion[emo] != []:
                # 添加最后一个
                data_new_set.append(data_emotion[emo][-1])
                index += 1
                # 舍去最后一个
                data_emotion[emo] = data_emotion[emo][0:-1]

if __name__ == "__main__":
    """
    clause_set: 语料库中的子句集合
    vocab_set: 语料库中的单词集合(及对应频数)
    tag_set: 二元组标签集合
    emotion_labels: 情绪词集合
    """
    clause_set = Vocabulary()
    vocab_set = Vocabulary()
    tag_set = Index()
    # other标签"O"
    tag_set.add("O")

    clause_set.load_clauses(PATH_CLAUSE_SET)
    vocab_set.load_words(PATH_VOCAB_SET)
    emotion_labels = Index()

    with open(PATH_DATA_SOURCE, 'rt', encoding='utf-8') as fin:
        dataset, overlap = prepare_data_set(fin, clause_set, vocab_set, tag_set, emotion_labels)
        data_save(dataset, PATH_DATA_SET)
    # print(overlap)# 475
    # print(len(dataset))# 1945
    emotion_labels.save(PATH_EMOTION_LABELS)
    tag_set.save(PATH_TAG_SET)

    data_set = data_load(PATH_DATA_SET)
    tag_set.load_words(PATH_TAG_SET)

    # ================== 定义字典,保存所有情感子句的数目(六种情绪 + 一个other) ==============================
    num_emotion = {}
    num_emotion['happiness'] = 0
    num_emotion['sadness'] = 0
    num_emotion['anger'] = 0
    num_emotion['disgust'] = 0
    num_emotion['fear'] = 0
    num_emotion['surprise'] = 0
    num_emotion['O'] = 0
    get_num_relaions(data_set, tag_set, num_emotion)
    print('=' * 118)
    print('num of different emotions')
    print('=' * 118)
    sum_all_emotion = 0
    '''
    happiness: 961
    sadness: 1025
    anger: 530
    disgust: 347
    fear: 705
    surprise: 149
    other: 24936
    '''
    for emo in num_emotion:
        print(emo,num_emotion[emo])
        sum_all_emotion += num_emotion[emo]
    sum_without_other = sum_all_emotion - num_emotion['O']
    # print(sum_all_emotion) # 28653
    # 各个关系类型的概率分别(包含'O'与不包含'O')
    print('=' * 118)
    print('rate of different emotions')
    print('=' * 118)
    '''
    happiness: 3.35%/25.85%
    sadness: 3.58%/27.58%
    anger: 1.85%/14.26%
    disgust: 1.21%/9.34%
    fear: 2.46%/18.97%
    surprise: 0.52%/4.01%
    other: 87.03%
    '''
    for emo in num_emotion:
        print(emo,"%.2f%%"%(num_emotion[emo]/sum_all_emotion*100))
        if emo != 'O':
            print('(without_other)',"%.2f%%"%(num_emotion[emo]/sum_without_other*100))
    # =============== 按不同关系类型整合训练数据，为交叉实验做准备 =========================
    data_emotion = {}
    data_emotion['happiness'] = []
    data_emotion['sadness'] = []
    data_emotion['anger'] = []
    data_emotion['disgust'] = []
    data_emotion['fear'] = []
    data_emotion['surprise'] = []
    # 按情绪类型归纳数据集
    get_data_emotions(data_set,data_emotion)
    # 保存对应数据集最大的情绪类型
    max_length = None
    total_length = 0
    for emo in data_emotion:
        length = len(data_emotion[emo])
        total_length += length
        if not max_length or max_length<length:
            max_length = length
    # print(total_length) # 1945
    # print(max_length) # 528

    # =================== 计算每一类情绪类型的抽取间隔 ====================================
    interval_emotion = {}
    interval_emotion['happiness'] = 0
    interval_emotion['sadness'] = 0
    interval_emotion['anger'] = 0
    interval_emotion['disgust'] = 0
    interval_emotion['fear'] = 0
    interval_emotion['surprise'] = 0

    # 以max_length为抽取基准，得到每个情绪的抽取间隔
    for emo in interval_emotion:
        interval_emotion[emo] = round(max_length/len(data_emotion[emo]))
        # print(emo,interval_emotion[emo])
    # print(len(data_emotion['happiness']))# 514
    # print(len(data_emotion['sadness']))# 528
    # print(len(data_emotion['anger']))# 266
    # print(len(data_emotion['disgust']))# 183
    # print(len(data_emotion['fear']))# 374
    # print(len(data_emotion['surprise']))# 80

    # ===================== 得到情感均匀分布的数据集 ========================================
    data_new_set = []
    get_new_dataset(data_new_set, interval_emotion, data_emotion, total_length)
    data_save(data_new_set, PATH_DATA_NEW_SET)
    # print(len(data_new_set)) # 1945