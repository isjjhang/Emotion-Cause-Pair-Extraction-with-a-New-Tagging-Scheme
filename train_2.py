"""
train_2.py:
    基于tagging.py中生成的data_set,
    实现六折交叉验证
"""
from CNN_CNN_LSTM_2 import CNN_CNN_LSTM
from CNN_BiLSTM_LSTM_2 import CNN_BiLSTM_LSTM
from utils_2 import *
import argparse
import bisect
import torch
from gensim.models import Doc2Vec
from gensim.models.word2vec import Word2Vec
from torch.utils.data.dataset import *
from torch.utils.data.sampler import *
from torch.nn.utils.rnn import *
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import time

# 设置六折交叉实验
DIVISION = 10
# 设置训练所用GPU
# cuda:0 对应x号GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
PATH_CLAUSE_SET = './data/clause_set.txt'
PATH_VOCAB_SET = './data/vocab_set.txt'
PATH_TAG_SET = './data/tag_set.txt'
PATH_EMOTION_LABELS = './data/emotion_labels.txt'
PATH_DATA_SET = './data/data_new_set.pk'
PATH_MODEL_CNN_BiLSTM = './model_2.pt'
PATH_MODEL_CNN_CNN = './model_3.pt'
# ================================ 设置模型相关参数 ====================================================================
parser = argparse.ArgumentParser(description='Joint Extraction of Entities and Relations Applying on ECR Task')

# ============== 模型参数设置 ======================
# 单词级卷积层数
parser.add_argument('--word_conv_layers', type=int, default=3)
# 子句级卷积层数
parser.add_argument('--clause_conv_layers', type=int, default=3)
# 词编码层卷积核大小
parser.add_argument('--word_kernel_size', type=int, default=3)
# 子句编码层卷积核大小
parser.add_argument('--clause_kernel_size', type=int, default=3)
# 每个单词级编码器的隐层单元数
parser.add_argument('--word_nhid', type=int, default=200)
# 每个子句级编码器的隐层单元数
parser.add_argument('--clause_nhid', type=int, default=300)
# dropout
parser.add_argument('--dropout', type=float, default=0.5)
# 除"other"标签外(初始权重=1)，每个标签在计算损失函数中的初始权重
parser.add_argument('--weight', type=float, default=10.0)

# ============== 训练参数设置 ===========================
# action='store_false': python train_2.py, args.cuda为 True
# python train_2.py --cuda, args.cuda为False
# action='store_true'相反
parser.add_argument('--cuda', action='store_false', help='use CUDA (default: True)')
# 一个epoch:过一遍训练集中的所有样本
parser.add_argument('--epochs', type=int, default=40)
# metavar:在 usage 说明中的参数名称，此处为"N"，默认为原变量名大写"BATCH_SIZE"
parser.add_argument('--batch_size', type=int, default=32, metavar='N')
parser.add_argument('--optim', type=str, default='SGD')
# 学习率
parser.add_argument('--lr', type=float, default=5)
# 梯度缩放因子
parser.add_argument('--clip', type=float, default=0.35)
# 输出当前训练状态的间隔
parser.add_argument('--report_interval', type=int, default=10, metavar='N')
# pytorch初始模型保存路径
parser.add_argument('--save_original', type=str, default='./org_model.pt')
# pytorch模型保存路径
parser.add_argument('--save_2', type=str, default=PATH_MODEL_CNN_BiLSTM)
parser.add_argument('--save_3', type=str, default=PATH_MODEL_CNN_CNN)
# 固定随机数种子，使结果确定，方便比较
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()
# 为CPU设置种子用于生成随机数，以使得结果是确定的，方便比较
torch.manual_seed(args.seed)
if args.cuda:
    # 为当前GPU设置随机种子
    torch.cuda.manual_seed(args.seed)
# 解决cudnn行为的不确定性问题
# torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device("cuda" if args.cuda else "cpu")

# ================================= 按day/hour/minute/second形式显示时间 ================================================
def time_display(s):
    # //:除后取整
    # day
    d = s // (3600*24)
    s -= d * (3600*24)
    # minute
    h = s // 3600
    s -= h * 3600
    # second
    m = s // 60
    s -= m * 60
    # format表示中^、<、>分别是居中、左对齐、右对齐，后面带宽度
    day_time = "{:1d}day ".format(int(d)) if d else "   "
    str_time = day_time + "{:0>2d}:{:0>2d}:{:0>2d}".format(int(h), int(m), int(s))
    return str_time

# ================================= 使用pickle.load(),解析文件中的二进制数据 ==========================================
def data_load(path): 
    # 其中file必须以二进制可读模式打开,即“rb”
    with open(path, 'rb') as f:
        return pickle.load(f)

# ================================= 将数据集依据start_ponits划分为训练集和测试集 ================================================
def get_divisions(data, start_points, idx, division=DIVISION):
    # idx从0开始
    train_data = []
    test_data = []

    length_data = len(data)
    start = start_points[idx]
    end = int(start + length_data/division)
    # 分别以start_points对应的组做测试集
    for i in range(0,start):
        train_data.append(data[i])
    for i in range(start, end):
        test_data.append(data[i])
    for i in range(end, length_data):
        train_data.append(data[i])
    
    return train_data, test_data

# ============================== 数据集处理 =====================================================================================
"""
clause_set: 语料库中的子句集合
vocab_set: 语料库中的单词及对应频数
tag_set: 二元组标签集合
emotion_labels: 情绪词集合
"""
clause_set = Vocabulary()
vocab_set = Vocabulary()
tag_set = Index()
emotion_labels = Index()

clause_set.load_clauses(PATH_CLAUSE_SET)
vocab_set.load_words(PATH_VOCAB_SET)
tag_set.load_words(PATH_TAG_SET)
emotion_labels.load_words(PATH_EMOTION_LABELS)
# 整理后的数据集
data_set = data_load(PATH_DATA_SET)
# print(len(clause_set)) # 27009 = 27007 + <pad> + <unk>
# print(len(vocab_set)) # 24096 = 24094 + <pad> + <unk>
# print(len(tag_set)) # 13
# print(len(emotion_labels)) # 6
# print(len(data_set)) # 1945

# 六折交叉验证
length_data = len(data_set)
# 获得每组数据的起始idx
start_points = []
for i in range(DIVISION):
    start_i = int(i*length_data/DIVISION)
    start_points.append(start_i)
# print(start_points) # [0, 194, 389, 583, 778, 972, 1167, 1361, 1556, 1750]

# 获得每组训练和测试数据
train_data = []
test_data = []
for i in range(DIVISION):
    train, test = get_divisions(data_set, start_points, i, DIVISION)
    train_data.append(train)
    test_data.append(test)

# word2vec:24094*200
word_embeddings = Word2Vec.load('./data/word2vec')
word_embeddings = torch.tensor(word_embeddings.wv.vectors)
word_embedding_size = word_embeddings.size(1) #200

# doc2vec:27007*300
clause_embeddings = Doc2Vec.load('./data/doc2vec')
clause_embeddings = torch.tensor(clause_embeddings.docvecs.vectors_docs)
clause_embedding_size = clause_embeddings.size(1) #300

# 词向量合并<pad><unk>
# 随机初始化两个tensor(<pad>与<unk>)并归一化至-0.01到0.01区间内
pad_embedding = torch.empty(1, word_embedding_size).uniform_(-0.01, 0.01)
unk_embedding = torch.empty(1, word_embedding_size).uniform_(-0.01, 0.01)
# 合并后24096*200
word_embeddings = torch.cat([pad_embedding, unk_embedding, word_embeddings])

# 子句向量合并<pad><unk>
pad_embedding = torch.empty(1, clause_embedding_size).uniform_(-0.01, 0.01)
unk_embedding = torch.empty(1, clause_embedding_size).uniform_(-0.01, 0.01)
# 合并后27009*300
clause_embeddings = torch.cat([pad_embedding, unk_embedding, clause_embeddings])

# [200, 200, 200, 200]
word_channels = [word_embedding_size] + [args.word_nhid] * args.word_conv_layers
# [500, 300, 300, 300]
# 将word特征与clause_embedding拼接起来作为clause encoder的输入
clause_channels = [clause_embedding_size + args.word_nhid] + [args.clause_nhid] * args.clause_conv_layers

# ==================== 创建训练模型 ============================================================================

'''
CNN_BiLSTM_LSTM:
    word_channels=[200, 200, 200, 200]
    word_kernel_size=3
    word_weight=word_embeddings(pad + unk + word2vec.wv.vectors)

    clause_embedding_size=300
    clause_weight=clause_embeddings(pad + unk + doc2vec.docvecs.vectors_docs)
    clause_hidden_dim:300(clause_feature_size=300*2=600)

    num_tag=len(tag_set): 13
    dropout: 0.5
'''
# to(device):将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
model=CNN_BiLSTM_LSTM(word_channels=word_channels, word_kernel_size=args.word_kernel_size, word_weight=word_embeddings,
            clause_embedding_size=clause_embedding_size, clause_weight=clause_embeddings, clause_hidden_dim = args.clause_nhid,
            num_tag=len(tag_set), dropout=args.dropout).to(device)

'''
CNN_CNN_LSTM:
    word_channels=[200, 200, 200, 200]
    word_kernel_size=3
    word_weight=word_embeddings(pad + unk + word2vec.wv.vectors)

    clause_embedding_size=300
    clause_channels=[500, 300, 300, 300]
    clause_kernel_size=3
    clause_weight=clause_embeddings(pad + unk + doc2vec.docvecs.vectors_docs)

    num_tag=len(tag_set): 13
    dropout: 0.5
'''
# model=CNN_CNN_LSTM(word_channels=word_channels, word_kernel_size=args.word_kernel_size, word_weight=word_embeddings,
#             clause_embedding_size=clause_embedding_size, clause_channels=clause_channels,
#             clause_kernel_size=args.clause_kernel_size, clause_weight=clause_embeddings, 
#             num_tag=len(tag_set), dropout=args.dropout).to(device)

# 为每个标签分配权重
# Other标签权重置1，其余为10
'''
weight:
    tensor([ 1., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
       device='cuda:0')
'''
weight = [args.weight] * len(tag_set)
weight[0] = 1
weight = torch.tensor(weight).to(device)

# 加权损失函数
# Negative Log Likelihood Loss(Cross-Entropy cost function)
# 针对前一步模型的输出(经过了log_softmax处理)，将每个单词与target上对应的标签概率进行累加
criterion = nn.NLLLoss(weight, reduction = 'sum')
# SGD优化
# getattr(optim, args.optim) = <class 'torch.optim.sgd.SGD'>
attr = getattr(optim, args.optim)
# 初始优化器
optimizer = attr(model.parameters(), lr=args.lr)

# 保存原始模型
with open(args.save_original, 'wb') as f:
    torch.save(model, f)

# ==================== 获得一个batch数据并padding ==================================================================
def get_batch(batch_index, data):
    # 按索引获取一个batch的数据
    batch = [data[idx] for idx in batch_index]
    # 按文本包含的子句数目降序排列(sorted不改变原序列)
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    # zip():对象中对应的元素打包成一个个元组
    # zip(* ):解压，返回元组中各个对象的二维矩阵形式
    doc, clause, tag = zip(*sorted_batch)
    tensor_doc = [torch.LongTensor(_) for _ in doc]
    tensor_clause = [torch.LongTensor(_) for _ in clause]
    tensor_tag = [torch.LongTensor(_) for _ in tag]
    # 数据对齐:先压缩后填充
    # pack_sequence():压缩不同长度的tensor列表，生成PackedSequence类
    # pad_packed_sequence():用padding_value填充回来; batch_first=True:将batch放在第一维
    padded_doc, lengths = pad_packed_sequence(pack_sequence(tensor_doc), batch_first=True, padding_value=0)
    padded_clause, _ = pad_packed_sequence(pack_sequence(tensor_clause), batch_first=True, padding_value=0)
    padded_tag, _ = pad_packed_sequence(pack_sequence(tensor_tag), batch_first=True, padding_value=0)
    return padded_doc.to(device), padded_clause.to(device), padded_tag.to(device), lengths.to(device)

# ================= 将文档数据依据所含子句数目划分至breakpoints序列中 ================================================
def get_groups(data, breakpoints):
    # 初始化 len(breakpoints)+1 个列表
    groups = [[] for _ in range(len(breakpoints)+1)]
    for idx, item in enumerate(data):
        # item[0]:doc_idx
        # len(item[0]):该doc包含clause个数
        # bisect_left():获得插入位置
        i = bisect.bisect_left(breakpoints, len(item[0]))
        groups[i].append(idx)
    # 按idx划分子集
    data_groups = [Subset(data, g) for g in groups]
    return data_groups

# ==================== 模型训练 ======================================================================================
def train(division_idx):
    # model.train():启用 BatchNormalization 和 Dropout
    model.train()
    # 当前所有节点损失值累加和
    train_loss = 0
    # 当前计算loss时的标签节点数
    train_count = 0
    # 所有节点总损失值
    total_loss = 0
    # 计算loss涉及的总标签节点数
    total_count = 0
    # 将训练样本按长度分为四组
    # 避免同一个batch内<pad>数过多
    breakpoints = [10, 20, 30]
    # 按文档的长度从大到小训练
    # 按doc中所含clause数目划分(MAX_DOC_LENGTH = 40)
    # 划分后的数据集含有对应原数据集的index
    train_data_groups = get_groups(train_data[division_idx], breakpoints)
    # 按batch_size(=32)对每一组内数据随机取样
    # 每组最后的数据可能不满足一个batch
    # len(sampler):总batch数
    sampler = GroupBatchRandomSampler(train_data_groups, args.batch_size, drop_last=False)
    for idx, batch_idx in enumerate(sampler):
        doc_idx, clause_idx, tag_idx, lengths = get_batch(batch_idx, train_data[division_idx])
        # 在backward每次计算梯度的时候，会将新的梯度值加到原来旧的梯度值上面
        # 将梯度清零（一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
        optimizer.zero_grad()
        # ====== 训练过程1：前向过程，计算输入到输出的结果 ==================
        # output: batch_size*num_clause*num_tag
        output = model(doc_idx, clause_idx)
        # pack_padded_sequence():确定长度，效果同pack_sequence
        # lengths:每条数据补0前对应长度
        output = pack_padded_sequence(output, lengths, batch_first=True).data
        tag_idx = pack_padded_sequence(tag_idx, lengths, batch_first=True).data
        # ====== 训练过程2：由结果和label计算损失 ==================
        loss = criterion(output, tag_idx)
        # ====== 训练过程3：在图的层次上面计算所有变量的梯度 ==================
        # 每次计算梯度的时候，其实是有一个动态的图在里面的，求导数就是对图中的参数w进行求导的过程
        # 每个参数计算的梯度值保存在w.grad.data上面，在参数更新时使用
        loss.backward()
        # 梯度裁剪
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # ============ 训练过程4：进行参数的更新 ==================
        # optimizer不计算梯度，它利用已经计算好的梯度值对参数进行更新
        optimizer.step()

        # 当前所有节点损失值累加和
        total_loss += loss.item()
        train_loss += loss.item()
        # len(tag_idx): 当前计算loss涉及的总标签数
        total_count += len(tag_idx)
        train_count += len(tag_idx)
        # 根据report_interval(100)输出
        if (idx+1) % args.report_interval == 0  or idx+1 == len(sampler):
            cur_loss = train_loss / train_count
            # 已训练时间
            elapsed = time.time() - start_time
            # 当前实验训练长度占总长度的百分比
            percent = ((epoch-1)*len(sampler)+(idx+1))/(args.epochs*len(sampler))
            # 剩余时间
            remaining = elapsed / percent - elapsed
            print("|Fold {:d} | Epoch {:2d}/{:2d} | Batch {:3d}/{:3d} | Elapsed Time {:s} | Remaining Time {:s} | "
                    "lr {:4.2e} | Loss {:5.4f} |".format((division_idx + 1), epoch, args.epochs, idx+1, len(sampler), 
                        time_display(elapsed), time_display(remaining), lr, cur_loss))
            train_loss = 0
            train_count = 0
    return total_loss/total_count

# ==================== 根据index获取抽取结果三元组 =======================================================================================     
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
        # role1:情绪子句, role2:原因子句
        role1, role2 = temp[emotion_label]
        # 二者皆不为空
        if role1 and role2:
            # 从情绪子句出发，同其对应原因子句配对
            for e1 in role1:
                for e2 in role2:
                    tuples.append((e1, emotion_label, e2))
                    # if (e1,e2) not in tuples:
                    #     tuples.append((e1, e2))
    return tuples
# ==================== 将生成的非空情绪原因对视为正样本，统计TP、TP+FP、TP+FN ==============================================================  
def measure(output, targets, lengths):
    # 当前batch_size,即measure的doc数目一致
    assert output.size(0) == targets.size(0) and targets.size(0) == lengths.size(0)
    # 预测为正样本中正确的部分,即正确预测的情绪原因对
    tp = 0
    # TP+FP:预测为正例的总样本数,即预测结果中的非空情绪原因对
    tp_fp = 0
    # TP+FN:实际正样本总数,即标签中的非空情绪原因对
    tp_fn = 0
    batch_size = output.size(0)
    # 获取每个word概率最大的tag_idx
    output = torch.argmax(output, dim=-1)

    for i in range(batch_size):
        length = lengths[i]
        # 获得output真实长度，即舍去<pad>部分
        # np.tolist():将数组矩阵转化为列表
        out = output[i][:length].tolist()
        out_tuples = get_tuples(out)
        # 预测为正例的总样本数
        tp_fp += len(out_tuples)

        target = targets[i][:length].tolist()
        target_tuples = get_tuples(target)
        # 实际正样本总数
        tp_fn += len(target_tuples)
        # 预测正确的正样本数
        for out_tuple in out_tuples:
            for target_tuple in target_tuples:
                if out_tuple == target_tuple:
                    tp += 1
    return tp, tp_fp, tp_fn

# ==================== 当前模型训练效果评估 ==================================================================
def evaluate(division_idx):
    # model.eval():不启用 BatchNormalization 和 Dropout
    model.eval()
    total_loss = 0
    total_count = 0
    # 预测为正样本中正确的部分
    TP = 0
    # 预测正
    TP_FP = 0
    # 实际正
    TP_FN = 0
    breakpoints = [10, 20, 30]
    test_data_groups = get_groups(test_data[division_idx], breakpoints)
    # 取消反向传播的自动求导
    with torch.no_grad():
        sampler = GroupBatchRandomSampler(test_data_groups, args.batch_size, drop_last=False)
        for batch_index in sampler:
            doc_idx, clause_idx, tag_idx, lengths = get_batch(batch_index, test_data[division_idx])
            output = model(doc_idx, clause_idx)
            tp, tp_fp, tp_fn = measure(output, tag_idx, lengths)
            TP += tp
            TP_FP += tp_fp
            TP_FN += tp_fn
            
            output = pack_padded_sequence(output, lengths, batch_first=True).data
            tag_idx = pack_padded_sequence(tag_idx, lengths, batch_first=True).data
            loss = criterion(output, tag_idx)
            total_loss += loss.item()
            total_count += len(tag_idx)
        # 防止除数为零
        if TP_FP == 0:
            precision = 1
        else:
            precision = TP/TP_FP
        if TP_FN == 0:
            recall = 1
        else:
            recall = TP/TP_FN
        if TP_FP == 0 and TP_FN == 0:
            f1 = 1
        else:
            f1 = 2*TP/(TP_FP + TP_FN)
    return total_loss / total_count, precision, recall, f1

if __name__ == "__main__":
    # 精确度(precision)：预测正例中正样本的比重
    all_precision = []
    # 召回率(recall)：正样本中预测为正例的比重
    all_recall = []
    # F1-Score：精确率和召回率的调和平均数
    all_f1 = []
    all_test_loss = []
    # 测试样本最佳损失值
    best_test_loss = None
    try:
        # ================ 六折交叉验证 =============================
        for division_idx in range(DIVISION):
            # 训练开始时间
            start_time = time.time()
            # 初始化学习率
            lr = args.lr
            # 训练样本最佳损失值
            best_train_loss = None

            # 加载初始化模型
            with open(args.save_original, 'rb') as f:
                model = torch.load(f)
            attr = getattr(optim, args.optim)
            # 初始优化器
            optimizer = attr(model.parameters(), lr=lr)
            # ============== 训练 ===================================
            print('=' * 118)
            print("Begin of Fold {:d}".format(division_idx + 1))
            print('=' * 118)

            # 按epoch训练
            for epoch in range(1, args.epochs+1):
                train_loss = train(division_idx)
                # 根据误差变换，降低学习率
                if not best_train_loss or train_loss < best_train_loss:
                    best_train_loss = train_loss
                else:
                    # 降低学习率
                    lr = lr / 5.0
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
            # ============== 测试 ===================================
            test_loss, precision, recall, f1 = evaluate(division_idx)
            all_test_loss.append(test_loss)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            print('=' * 118)
            print("| End of Fold {:d} | Test Loss {:5.4f} | Precision {:5.4f} | Recall {:5.4f} | "
                "F1 {:5.4f} |".format((division_idx + 1), test_loss, precision, recall, f1))
            
            # 更新当前最优模型并保存
            if not best_test_loss or test_loss < best_test_loss:
                with open(args.save_2, 'wb') as f:
                   torch.save(model, f)
                # with open(args.save_3, 'wb') as f:
                #     torch.save(model, f)
                best_test_loss = test_loss
    # ctrl+c终止
    except KeyboardInterrupt:
        print('=' * 118)
        print('Exiting from training early')
        print('=' * 118)
    # ============ 求六折交叉实验均值 ========================
    avg_test_loss = np.mean(all_test_loss)
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)
    print('=' * 118)
    print('=' * 118)
    print("| Average | Test Loss {:5.4f} | Precision {:5.4f} | Recall {:5.4f} | "
        "F1 {:5.4f} |".format(avg_test_loss, avg_precision, avg_recall, avg_f1))
    with open("./data/record.tsv", "wt", encoding="utf-8") as f:
        f.write("Index\ttest_loss\tprecision\trecall\tF1-Score\n")
        for idx in range(len(all_test_loss)):
            f.write("{:d}\t{:5.4f}\t{:5.4f}\t{:5.4f}\t{:5.4f}\n".format(idx+1, all_test_loss[idx], all_precision[idx], all_recall[idx], all_f1[idx]))
        f.write("Average\t{:5.4f}\t{:5.4f}\t{:5.4f}\t{:5.4f}\n".format(avg_test_loss, avg_precision, avg_recall, avg_f1))