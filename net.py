import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from game import Engine


class SnakeNet(nn.Module):
    def __init__(self):
        super(SnakeNet, self).__init__()

        self.actions_num = 3
        self.gamma = 0.9
        self.final_epsilon = .0001
        self.init_epsilon = .5
        self.iterations_num = 100000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.fc1 = nn.Linear(11, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, self.actions_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)

        return x


def create_torch_state(state):
    state = torch.from_numpy(state).unsqueeze(0).float()

    return state.to('cuda')


if __name__ == '__main__':
    wandb.init(project='snake-dql', name='manual-extraction-v8')

    game = Engine(board_size=20)
    model = SnakeNet().cuda()
    wandb.watch(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    replay_memory = []
    epsilon = model.init_epsilon

    board_state = game.get_board_state()
    state = create_torch_state(board_state)

    epsilon_delta = (model.init_epsilon -
                     model.final_epsilon) / model.iterations_num

    max_reward = 0
    games_played = 0
    last_reward = 0

    model.train()
    for i in range(model.iterations_num):
        output = model(state)[0]
        max_q = torch.max(output).item()

        action = torch.zeros([model.actions_num], dtype=torch.float32)
        action = action.to('cuda')

        random_action = random.random() <= epsilon
        action_idx = random.randint(
            0, 2) if random_action else torch.argmax(output).item()
        action[action_idx] = 1
        action_game = action_idx - 1

        board_state, reward, terminal = game.next_round_nn(action_game)
        max_reward = max(max_reward, reward)
        tmp_reward = reward
        reward -= last_reward
        last_reward = tmp_reward
        new_state = create_torch_state(board_state)

        if terminal:
            game.reset()
            games_played += 1

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
            print(f'Iteration: {i//1000}/{model.iterations_num // 1000}')
            wandb.log({'Max Q value': max_q, 'Loss': loss,
                       'Max reward': max_reward, 'Games': games_played})

        loss.backward()
        optimizer.step()

        state = new_state

    model_state = {
        'model_state_dict': model.state_dict(),
        'replay_memory': replay_memory
    }

    torch.save(model_state, os.path.join(wandb.run.dir, 'model.pt'))
