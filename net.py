import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from game import Engine


class SnakeNet(nn.Module):
    def __init__(self):
        super(SnakeNet, self).__init__()

        self.actions_num = 3
        self.gamma = 0.99
        self.final_epsilon = .0001
        self.init_epsilon = .9
        self.iterations_num = 100000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(2, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc3 = nn.Linear(32 * 8 * 8, 128)
        self.fc4 = nn.Linear(128, self.actions_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def create_state(last_board, board):
    np_state = np.transpose(np.dstack((last_board, board)), (2, 1, 0))
    state = torch.from_numpy(np_state)
    state = state.unsqueeze(0).float()

    return state.to('cuda')


if __name__ == '__main__':
    game = Engine(board_size=10)
    model = SnakeNet().cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()

    replay_memory = []
    epsilon = model.init_epsilon

    board = game.to_numpy()
    last_board = board
    state = create_state(last_board, board)

    epsilon_delta = (model.init_epsilon -
                     model.final_epsilon) / model.iterations_num

    for i in range(model.iterations_num):
        output = model(state)[0]

        action = torch.zeros([model.actions_num], dtype=torch.float32)
        action = action.to('cuda')

        random_action = random.random() <= epsilon
        action_idx = random.randint(
            0, 2) if random_action else torch.argmax(output).item()
        action[action_idx] = 1
        action_game = action_idx - 1

        last_board = board
        board, reward, terminal = game.next_round_nn(action_game)
        new_state = create_state(last_board, board)

        if terminal:
            game = Engine(board_size=10)

        reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)

        action = action.unsqueeze(0)
        replay_memory.append((state, action, reward, new_state, terminal))

        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        epsilon -= epsilon_delta

        minibatch = random.sample(replay_memory, min(
            len(replay_memory), model.minibatch_size))

        state_batch = torch.cat(tuple(d[0] for d in minibatch)).to('cuda')
        action_batch = torch.cat(tuple(d[1] for d in minibatch)).to('cuda')
        reward_batch = torch.cat(tuple(d[2] for d in minibatch)).to('cuda')
        new_state_batch = torch.cat(tuple(d[3] for d in minibatch)).to('cuda')

        new_state_batch_output = model(new_state_batch)

        y_batch = torch.cat(tuple(reward_batch[j] if minibatch[j][4] else reward_batch[j] +
                                  model.gamma * torch.max(new_state_batch_output[j]) for j in range(len(minibatch))))

        q_vals = torch.sum(model(state_batch) * action_batch, dim=1)

        optimizer.zero_grad()

        y_batch.detach_()

        loss = criterion(q_vals, y_batch)
        if i % 1000 == 0:
            print(f'Iteration: {i/1000}/100')

        loss.backward()
        optimizer.step()

        state = new_state

    torch.save(model.state_dict(), './snake_model.pt')
