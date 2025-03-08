import torch.nn as nn
import torch.nn.functional as F


class CaptchaModelCNN(nn.Module):
    def __init__(self):
        super(CaptchaModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (7, 11), stride=(1, 2))
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 10 * 15, 256)
        self.fc2 = nn.Linear(256, 5 * 36)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        # print(x.shape)
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
