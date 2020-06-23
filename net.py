import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game import Engine


class SnakeNet(nn.Module):
    def __init__(self):
        super(SnakeNet, self).__init__()

        self.actions_num = 3
        self.gamma = 0.99
        self.final_epsilon = .0001
        self.init_epsilon = .5
        self.iterations_num = 100000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(2, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc3 = nn.Linear(32 * 8 * 8, 128)
        self.fc4 = nn.Linear(128, self.actions_num)

    def forward(self, x):
        x = F.relu(self.con1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


model = SnakeNet()
optimizer = optim.Adam(model.parameters(), lr=1e-6)
criterion = nn.MSELoss()
replay_memory = []
game = Engine(board_size=10)
