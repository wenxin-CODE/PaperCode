import torch
import torch.nn as nn
from collections import OrderedDict
import math
import torch.nn.functional as F

class Att(torch.nn.Module):
    def __init__(self, channels=None, out_channels=None, e_lambda=1e-4):
        super(Att, self).__init__()
        self.activaton = nn.Sigmoid()
        # self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    def forward(self, x):
        # b, c = x.size()
        # n = 128  #此处使用的是batch_size
        # x_minus_mu_square = (x - x.mean(dim=1, keepdim=True)).pow(2)
        # y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=1, keepdim=True) / n + self.e_lambda)) + 0.5

        return self.activaton(x)

class SleepEEGNet(nn.Module):
    def __init__(self, config):
        super(SleepEEGNet, self).__init__()
        self.padding_edf = {  # same padding in tensorflow
            'conv1': (22, 22),
            'max_pool1': (2, 2),
            'conv2': (3, 4),
            'max_pool2': (0, 1),
        }
        self.config = config
        first_filter_size = int(self.config["sampling_rate"] / 2.0)  # 100/2 = 50, 与以往使用的Resnet相比，这里的卷积核更大
        first_filter_stride = int(self.config["sampling_rate"] / 16.0)  # todo 与论文不同，论文给出的stride是100/4=25
        self.cnn = nn.Sequential(
            nn.ConstantPad1d(self.padding_edf['conv1'], 0),  # conv1
            nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels=1, out_channels=128, kernel_size=first_filter_size, stride=first_filter_stride,
                      bias=False))
            ])),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool1'], 0),  # max p 1
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv2
            nn.Sequential(OrderedDict([
                ('conv2',
                 nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),

            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv3
            nn.Sequential(OrderedDict([
                ('conv3',nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),

            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv4
            nn.Sequential(OrderedDict([
                ('conv4', nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool2'], 0),  # max p 2
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=0.5),
        )

        self.cnn2 = nn.Sequential(
            nn.ConstantPad1d(self.padding_edf['conv1'], 0),  # conv1
            nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels=1, out_channels=128, kernel_size=200, stride=50,
                      bias=False))
            ])),
            # nn.BatchNorm1d(128),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool1'], 0),  # max p 1
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv2
            nn.Sequential(OrderedDict([
                ('conv2',
                 nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, bias=False))
            ])),
            # nn.BatchNorm1d(128),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),

            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv3
            nn.Sequential(OrderedDict([
                ('conv3',nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, bias=False))
            ])),
            # nn.BatchNorm1d(128),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),

            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv4
            nn.Sequential(OrderedDict([
                ('conv4', nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, bias=False))
            ])),
            # nn.BatchNorm1d(128),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool2'], 0),  # max p 2
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=0.5),
        )
        # self.rnn = nn.LSTM(input_size=2048, hidden_size=self.config['n_rnn_units'], num_layers=1, dropout=0.5)
        # self.rnn = nn.LSTM(input_size=2048, hidden_size=self.config['n_rnn_units'], num_layers=1)
        self.rnn = nn.LSTM(input_size=2048, hidden_size=self.config['n_rnn_units'], num_layers=1, batch_first=True)
        self.att = Att()
        self.rnn2 = nn.LSTM(input_size=128, hidden_size=self.config['n_rnn_units'], num_layers=1, batch_first=True)
        self.rnn_dropout = nn.Dropout(p=0.5)  # todo 是否需要这个dropout?
        self.fc1 = nn.Linear(3040,2560)
        self.fc2 = nn.Linear(self.config['n_rnn_units'], 5)


    # for SleepEEGNet
    def forward(self, x, state):
        # x = torch.cat((self.cnn(x),self.cnn2(x)),1)
        x1 = self.cnn(x)
        x2 = self.cnn2(x)


        # Padding should be (left, right, top, bottom) for each dimension
        # Since we want to pad evenly on both sides, we split the padding into two parts
        padding = (1,1663)

        # Pad the tensor
        x2 = F.pad(x2, padding, mode='constant', value=0)
        x = x1+x2
        # x = self.cnn(x)+self.cnn2(x)
        x = x.view(-1, self.config['seq_length'], 2048)  # batch first == True
        assert x.shape[-1] == 2048
        x, state = self.rnn(x, state)
        x = x*self.att(x)
        x, state = self.rnn2(x, state)
        # x = x.view(-1, self.config['n_rnn_units'])
        x = x.reshape(-1, self.config['n_rnn_units'])
        # rnn output shape(seq_length, batch_size, hidden_size)
        x = self.rnn_dropout(x)
        # x = self.fc1(x.T)
        x = self.fc2(x)

        return x, state

    # def __init__(self, n_classes=5, channels=1, samples=3000,
    #              dropoutRate=0.5, kernelLength=64, kernelLength2=16,
    #              F1=8, D=2, F2=16,config):
    #     super(TinySleepNet, self).__init__()
    #     self.F1 = F1
    #     self.F2 = F2
    #     self.D = D
    #     self.samples = samples
    #     self.n_classes = n_classes
    #     self.channels = channels
    #     self.kernelLength = kernelLength
    #     self.kernelLength2 = kernelLength2
    #     self.drop_out = dropoutRate
    #     self.config = config

    #     block1 = nn.Sequential(
    #         # nn.ZeroPad2d([31, 32, 0, 0]), # Pads the input tensor boundaries with zero. [left, right, top, bottom]
    #         # input shape (1, C, T)
    #         nn.Conv2d(
    #             in_channels=1,
    #             out_channels=self.F1, # F1
    #             kernel_size=(1, self.kernelLength), # (1, half the sampling rate)
    #             stride=1,
    #             padding=(0, self.kernelLength//2),
    #             bias=False
    #         ), # output shape (F1, C, T)
    #         nn.BatchNorm2d(num_features=self.F1)
    #         # output shape (F1, C, T)
    #     )

    #     block2 = nn.Sequential(
    #         # input shape (F1, C, T)
    #         # Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, stride=1, padding=(0, 0),
    #         #                      groups=self.F1, bias=False),
    #         nn.Conv2d(
    #             in_channels=self.F1,
    #             out_channels=self.F1*self.D, # D*F1
    #             kernel_size=(self.channels, 1), # (C, 1)
    #             groups=self.F1, # When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also known as a “depthwise convolution”.
    #             bias=False
    #         ), # output shape (self.F1*self.D, 1, T)
    #         nn.BatchNorm2d(num_features=self.F1*self.D),
    #         nn.ELU(),
    #         nn.AvgPool2d(
    #             kernel_size=(1, 4),
    #             stride=4,), # output shape (self.F1*self.D, 1, T/4)
    #         nn.Dropout(p=self.drop_out)
    #         # output shape (self.F1*self.D, 1, T/4)
    #     )

    #     block3 = nn.Sequential(
    #         # nn.ZeroPad2d((7, 8, 0, 0)),
    #         # input shape (self.F1*self.D, 1, T/4)
    #         nn.Conv2d(
    #             in_channels=self.F2,
    #             out_channels=self.F2, # F2 = D*F1
    #             kernel_size=(1, self.kernelLength2),
    #             stride=1,
    #             padding=(0, self.kernelLength2//2),
    #             groups=self.F1*self.D,
    #             bias=False
    #         ), # output shape (self.F2, 1, T/4)
    #         # input shape (self.F2, 1, T/4)
    #         nn.Conv2d(
    #             in_channels=self.F1*self.D,
    #             out_channels=self.F2, # F2 = D*F1
    #             kernel_size=(1, 1),
    #             stride=1,
    #             bias=False
    #         ), # output shape (self.F2, 1, T/4)
    #         nn.BatchNorm2d(num_features=self.F2),
    #         nn.ELU(),
    #         nn.AvgPool2d(
    #             kernel_size=(1, 8),
    #             stride=8),  # output shape (self.F2, 1, T/4/8)
    #         nn.Dropout(p=self.drop_out)
    #         # output shape (self.F2, 1, T/32)
    #     )

    #     self.EEGNetLayer = nn.Sequential(block1, block2, block3)
    #     self.rnn = nn.LSTM(input_size=2048, hidden_size=self.config['n_rnn_units'], num_layers=1, batch_first=True)
    #     self.rnn_dropout = nn.Dropout(p=0.5)  # todo 是否需要这个dropout?
    #     self.fc = nn.Linear(self.config['n_rnn_units'], 5)

    # def forward(self, x, state):
    #     if len(x.shape) is not 4:
    #         x = torch.unsqueeze(x, 1)
    #     x = self.EEGNetLayer(x)
    #     x = x.view(-1, self.config['seq_length'], 2048)  # batch first == True
    #     assert x.shape[-1] == 2048
    #     x, state = self.rnn(x, state)
    #     # x = x.view(-1, self.config['n_rnn_units'])
    #     x = x.reshape(-1, self.config['n_rnn_units'])
    #     # rnn output shape(seq_length, batch_size, hidden_size)
    #     x = self.rnn_dropout(x)
    #     x = self.fc(x)

    #     return x, state


if __name__ == '__main__':
    from torchsummaryX import summary
    from config.sleepedf import train

    model = TinySleepNet(config=train)
    state = (torch.zeros(size=(1, 2, 128)),
             torch.zeros(size=(1, 2, 128)))
    torch
    # state = (state[0].to(self.device), state[1].to(self.device))
    summary(model, torch.randn(size=(2, 1, 3000)), state)



