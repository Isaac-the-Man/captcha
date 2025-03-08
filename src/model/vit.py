import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=18):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class CaptchaModelViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), padding='same')
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, (3, 3), stride=(2, 1), padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, (3, 3), padding='same')
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(256, 512, (3, 3))
        self.pe = PositionalEncoding(512, max_len=18)
        self.transformerLayer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformerEncoder = nn.TransformerEncoder(self.transformerLayer, num_layers=3)
        self.fc = nn.Linear(512, 36 + 1) # 36 alphanumeric characters + 1 blank

    def forward(self, x):
        # print(f'{x.shape=}')
        x = F.relu(self.conv1(x))
        # print(f'{x.shape=}')
        x = self.pool1(x)
        # print(f'{x.shape=}')
        x = F.relu(self.conv2(x))
        # print(f'{x.shape=}')
        x = self.pool2(x)
        # print(f'{x.shape=}')
        x = F.relu(self.conv3(x))
        # print(f'{x.shape=}')
        x = self.pool3(x)
        # print(f'{x.shape=}')
        x = F.relu(self.conv4(x)) # (batch, 512, 1, 18)
        # print(f'{x.shape=}')

        # transpose to (batch, seq, feature)
        x = x.permute(0, 3, 1, 2).squeeze(3) # (batch, 18, 512)
        # print(f'{x.shape=}')

        x = self.pe(x)
        x = self.transformerEncoder(x) # (batch, 18, 512)
        # print(f'{x.shape=}')
        # print(f'{h.shape=}')

        x = self.fc(x) # (batch, 18, 37)
        # print(f'{x.shape=}')

        return x

if __name__ == '__main__':
    model = CaptchaModelViT()
    x = torch.randn(1024, 1, 60, 160)
    y = model(x)
    print(y.shape)