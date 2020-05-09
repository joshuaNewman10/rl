import torch.nn as nn
import torch.nn.functional as F


class DQCNN(nn.Module):
    def __init__(self, action_size: int):
        super(DQCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2)
        self.fc1 = nn.Linear(in_features=32, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc2(x))

        return y
