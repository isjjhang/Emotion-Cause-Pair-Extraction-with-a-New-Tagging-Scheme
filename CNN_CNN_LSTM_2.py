"""
CNN_CNN_LSTM_2.py:
    构造CNN-CNN-LSTM模型
    类WordEncoder: 单词级CNN编码器
    类ClauseEncoder: 子句级CNN编码器
    类Decoder: LSTM解码器
    类CNN_CNN_LSTM: 拼接整合模型
"""

from conv_net import ConvNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.models import Doc2Vec

# 词级编码层
class WordEncoder(nn.Module):
    """
    Input: (total_clause_num, clause_len)
    Output: (total_clause_num, word_conv_size)
    """
    def __init__(self, word_weight, word_channels, word_kernel_size, dropout):
        super(WordEncoder, self).__init__()
        self.word_weight = word_weight
        self.word_channels = word_channels
        self.word_kernel_size = word_kernel_size
        # 获取word2vec词向量
        # freeze=False:继续更新权重
        self.embed = nn.Embedding.from_pretrained(word_weight, freeze=False)
        self.conv_net = ConvNet(word_channels, word_kernel_size, dropout)

    def forward(self, inputs):
        # clause_len = MAX_CLAUSE_LENGTH = 20
        clause_len = inputs.size(1)
        # (total_clause_num, clause_len) -> (total_clause_num, clause_len, word_embedding_size)
        embeddings = self.embed(inputs)
        # -> (total_clause_num, word_embedding_size, clause_len)
        # contiguous(): 把tensor变成在内存中连续分布的形式
        trans = embeddings.transpose(1, 2).contiguous()
        # -> (total_clause_num, word_conv_size, clause_len)
        conv = self.conv_net(trans)
        # -> (total_clause_num, word_conv_size, 1)
        # ( torch.nn.functional.max_pool1d(input, kernel_size) )
        max_pool = F.max_pool1d(conv, clause_len)
        # -> (total_clause_num, word_conv_size)
        # squeeze():舍弃维度为1的那一维
        sque = max_pool.squeeze()
        return sque

# 子句编码层
class ClauseEncoder(nn.Module):
    """
    Input: clause_input:(batch_size, clause_num), word_input:(batch_size, clause_num, word_conv_size)
    Output: (batch_size, clause_num, word_conv_size + clause_embedding_size + clause_conv_size)
    """
    def __init__(self, clause_weight, clause_channels, clause_kernel_size, dropout):
        super(ClauseEncoder, self).__init__()
        self.clause_weight = clause_weight
        self.clause_channels = clause_channels
        self.clause_kernel_size = clause_kernel_size
        self.dropout = dropout
        # 获取doc2vec子句向量
        # freeze=False:继续更新权重
        self.embed = nn.Embedding.from_pretrained(clause_weight, freeze=False)
        self.conv_net = ConvNet(clause_channels, clause_kernel_size, dropout, dilated=True, residual=False)

    def forward(self, clause_input, word_input):
        # (batch_size, clause_num) -> (batch_size, clause_num, clause_embedding_size)
        embeddings = self.embed(clause_input)
        # -> (batch_size, clause_num, clause_embedding_size + word_conv_size)
        cat = torch.cat((embeddings, word_input), 2)
        # -> (batch_size, clause_embedding_size + word_conv_size, clause_num)
        trans = cat.transpose(1, 2).contiguous()
        # -> (batch_size, clause_conv_size, clause_num)
        conv_out = self.conv_net(trans)
        # -> (batch_size, clause_conv_size + clause_embedding_size + word_conv_size, clause_num)
        cat2 = torch.cat((trans, conv_out), 1)
        # -> (batch_size, clause_num, clause_conv_size + clause_embedding_size + word_conv_size)
        trans2 = cat2.transpose(1, 2).contiguous()
        return trans2

# LSTM解码层
class Decoder(nn.Module):
    """
    Input: (batch_size, clause_num, clause_conv_size + clause_embedding_size + word_conv_size)
    Output: (batch_size, clause_num, num_tag)

    """
    def __init__(self, input_size, hidden_dim, output_size, NUM_LAYERS):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        # num_layers:隐藏层层数
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first = True, num_layers = NUM_LAYERS)
        self.hidden2label = nn.Linear(hidden_dim, output_size)
        self.init_weight()

    def forward(self, inputs):
        # flatten_parameters: 使得parameter的数据存放成contiguous chunk(连续的块)
        self.lstm.flatten_parameters()
        #  -> (batch_size, clause_num, hidden_dim)
        lstm_out, self.hidden = self.lstm(inputs, None)
        #  -> (batch_size, clause_num, num_tag)
        y = self.hidden2label(lstm_out)
        return y

    def init_weight(self):
        nn.init.kaiming_uniform_(self.hidden2label.weight.data, mode='fan_in', nonlinearity='relu')

class CNN_CNN_LSTM(nn.Module):
    """
    训练模型
    """
    def __init__(self, word_channels, word_kernel_size, word_weight,
            clause_embedding_size, clause_channels, clause_kernel_size, clause_weight,
            num_tag, dropout):
        super(CNN_CNN_LSTM, self).__init__()
        self.word_channels = word_channels
        self.word_kernel_size = word_kernel_size
        self.word_weight = word_weight
        self.clause_embedding_size = clause_embedding_size
        self.clause_channels = clause_channels
        self.clause_kernel_size = clause_kernel_size
        self.clause_weight = clause_weight
        self.num_tag = num_tag

        # 单词编码层
        self.word_encoder = WordEncoder(word_weight, word_channels, word_kernel_size, dropout)
        # 子句编码层
        self.clause_encoder = ClauseEncoder(clause_weight, clause_channels, clause_kernel_size, dropout)
        # 单词卷积后通道数
        self.word_conv_size = word_channels[-1]
        self.clause_conv_size = clause_channels[-1]
        # 子句卷积后通道数(200+300+300)
        self.decoder = Decoder(self.word_conv_size + self.clause_embedding_size + self.clause_conv_size,
                               self.word_conv_size + self.clause_embedding_size + self.clause_conv_size,
                               num_tag, NUM_LAYERS=1)

    def forward(self, clause_input, word_input):
        batch_size = clause_input.size(0)
        # 每个文档中子句数目
        clause_num = clause_input.size(1)
        # 词编码层
        # -1表示维数自动判断
        # word_input: (batch_size*clause_num*MAX_CLAUSE_LENGTH(=20))
        #           单个数据对应vocab_set中的单个词idx
        word_input = word_input.view(-1, word_input.size(2))
        word_output = self.word_encoder(word_input)
        word_output = word_output.view(batch_size, clause_num, -1)
        # 子句编码层
        # clause_input: (batch_size*clause_num)
        #           单个数据对应clause_set中的单个子句idx
        clause_output = self.clause_encoder(clause_input, word_output)
        # LSTM解码层
        y = self.decoder(clause_output)
        # 按num_tag所在维度取对数softmax
        return F.log_softmax(y, dim=2)

if __name__ == "__main__":
    # word_embedding_size=200
    word_embeddings = Word2Vec.load('./data/word2vec')
    word_embeddings = torch.tensor(word_embeddings.wv.vectors)

    # clause_embedding_size=300
    clause_embeddings = Doc2Vec.load('./data/doc2vec')
    clause_embeddings = torch.tensor(clause_embeddings.docvecs.vectors_docs)
    model=CNN_CNN_LSTM(word_channels=[200, 200, 200, 200], word_kernel_size=3, word_weight=word_embeddings,
              clause_embedding_size=300, clause_channels=[500, 300, 300, 300], clause_kernel_size=3, clause_weight=clause_embeddings, 
              num_tag=13, dropout=0.5)
    print(model)
'''
CNN_CNN_LSTM(
  (word_encoder): WordEncoder(
    (embed): Embedding(24094, 200)
    (conv_net): ConvNet(
      (net): Sequential(
        (0): Dropout(p=0.5)
        (1): ConvBlock(
          (conv): Conv1d(200, 200, kernel_size=(3,), stride=(1,), padding=(1,))
          (activate): ReLU()
        )
        (2): Dropout(p=0.5)
        (3): ConvBlock(
          (conv): Conv1d(200, 200, kernel_size=(3,), stride=(1,), padding=(1,))
          (activate): ReLU()
        )
        (4): Dropout(p=0.5)
        (5): ConvBlock(
          (conv): Conv1d(200, 200, kernel_size=(3,), stride=(1,), padding=(1,))
          (activate): ReLU()
        )
      )
    )
  )
  (clause_encoder): ClauseEncoder(
    (embed): Embedding(27007, 300)
    (conv_net): ConvNet(
      (net): Sequential(
        (0): Dropout(p=0.5)
        (1): ConvBlock(
          (conv): Conv1d(500, 300, kernel_size=(3,), stride=(1,), padding=(1,))
          (activate): ReLU()
        )
        (2): Dropout(p=0.5)
        (3): ConvBlock(
          (conv): Conv1d(300, 300, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
          (activate): ReLU()
        )
        (4): Dropout(p=0.5)
        (5): ConvBlock(
          (conv): Conv1d(300, 300, kernel_size=(3,), stride=(1,), padding=(3,), dilation=(3,))
          (activate): ReLU()
        )
      )
    )
  )
  (drop): Dropout(p=0.5)
  (decoder): Decoder(
    (lstm): LSTM(800, 800, batch_first=True)
    (hidden2label): Linear(in_features=800, out_features=13, bias=True)
  )
)
'''