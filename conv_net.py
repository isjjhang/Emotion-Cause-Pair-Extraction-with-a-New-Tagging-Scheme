"""
conv_net.py:
    构造卷积单元
    类ConvBlock: 构造带有残差结构的卷积单元
    类ConvNet: 根据输入参数构造多层卷积，并增加Dropout层
"""

import torch.nn as nn
from torch.nn.utils import weight_norm

# 卷积单元：F(x) = Conv1d + ReLu + x/down_sample(x) 
class ConvBlock(nn.Module):
    """
    一个卷积单元
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, residual=True):
        super(ConvBlock, self).__init__()
        # stride: 卷积步长
        # weight_norm: 应用权重标准化
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True))
        self.activate = nn.ReLU()
        self.residual = residual
        # 下采样，用于残差结构中解决F(x)与x维度不一致问题
        self.down_sample = nn.Conv1d(in_channels, out_channels, 1) if residual and in_channels != out_channels else None
        self.init_weights()

    # ===== 初始化权重参数：He initialization 均匀分布==========
    def init_weights(self):
        # 针对Relu,用kaiming初始化卷积层参数
        nn.init.kaiming_uniform_(self.conv.weight.data, nonlinearity='relu')
        if self.conv.bias is not None:
            self.conv.bias.data.fill_(0)
        if self.down_sample is not None:
            nn.init.kaiming_uniform_(self.down_sample.weight.data, nonlinearity='relu')
            if self.down_sample.bias is not None:
                self.down_sample.bias.data.fill_(0)

    # ======== 输入前向传播 =============
    def forward(self, inputs):
        output = self.activate(self.conv(inputs))
        if self.residual:
            output += self.down_sample(inputs) if self.down_sample else inputs
        return output

class ConvNet(nn.Module):
    """
    添加Dropout层后的多层卷积网络
    """
    def __init__(self, channels, kernel_size=3, dropout=0.5, dilated=False, residual=True):
        super(ConvNet, self).__init__()
        # 按卷积层数设置每层输入输出通道数
        num_levels = len(channels)-1
        layers = []
        for i in range(num_levels):
            in_channels = channels[i]
            out_channels = channels[i+1]
            padding = (kernel_size - 1) // 2
            layers += [nn.Dropout(dropout), ConvBlock(in_channels, out_channels, kernel_size, padding=padding, residual=residual)]
        self.net = nn.Sequential(*layers)
    def forward(self, inputs):
        return self.net(inputs)

if __name__ == "__main__":
    # =========================== test =============================================
    conv = ConvNet([200,200,400,400])
    print(conv.net)
    '''
    # ========================== test_output ======================================
    Sequential(
    (0): ConvBlock(
        (conv): Conv1d(200, 200, kernel_size=(3,), stride=(1,), padding=(1,))
        (activate): ReLU()
    )
    (1): Dropout(p=0.5)
    (2): ConvBlock(
        (conv): Conv1d(200, 400, kernel_size=(3,), stride=(1,), padding=(1,))
        (activate): ReLU()
        (down_sample): Conv1d(200, 400, kernel_size=(1,), stride=(1,))
    )
    (3): Dropout(p=0.5)
    (4): ConvBlock(
        (conv): Conv1d(400, 400, kernel_size=(3,), stride=(1,), padding=(1,))
        (activate): ReLU()
    )
    )
    '''