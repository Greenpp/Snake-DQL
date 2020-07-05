import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.core.decorators import auto_move_data

import wandb
from game import Engine


class Memory(IterableDataset):
    def __init__(self, size, sample_size, iterations):
        super(Memory).__init__()
        self.max_size = size
        self.sample_size = sample_size
        self.buffer = []
        self.iterations = iterations

    def __iter__(self):
        # buff_size = len(self.buffer)
        # sample_size = min(self.sample_size, buff_size)
        # sample = random.sample(self.buffer, sample_size) if self.it < self.iterations else []
        # for s in sample:
        #     self.it += 1
        #     yield s
        while True:
            yield self.buffer[random.randint(0, len(self.buffer) - 1)]

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)


class Agent:
    def __init__(self, batch_size, memory_size, init_epsilon, end_epsilon, iterations_num, actions_num):
        self.game_engine = Engine(board_size=20)
        self.replay_memory = Memory(memory_size, batch_size, iterations_num)

        self.epsilon_delta = (init_epsilon - end_epsilon) / iterations_num
        self.batch_size = batch_size
        self.epsilon = init_epsilon
        self.actions_num = actions_num
        state = self.game_engine.get_board_state()
        self.game_state = torch.from_numpy(state)

    def move(self, model_output):
        random_action = model_output is None or random.random() <= self.epsilon
        action_idx = random.randint(0, self.actions_num - 1) if random_action else torch.argmax(model_output).item()
        action = torch.zeros(self.actions_num)
        action[action_idx] = 1

        new_state, reward, terminal = self.game_engine.next_round_nn(action_idx)
        new_state = torch.from_numpy(new_state)

        self.replay_memory.append((self.game_state, action, reward, new_state, terminal))

        self.game_state = new_state
        self.epsilon -= self.epsilon_delta

        if terminal:
            self.game_engine.reset()

    def get_board_state(self):
        return self.game_state

class SnakeNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.actions_num = 3
        self.gamma = 0.9
        self.final_epsilon = .0001
        self.init_epsilon = .5
        self.iterations_num = 1000
        self.replay_memory_size = 10000
        self.batch_size = 32
        self.cur_batch_size = 0

        self.agent = Agent(self.batch_size, self.replay_memory_size, self.init_epsilon,
                           self.final_epsilon, self.iterations_num, self.actions_num)
        self.it_num = 0

        self.fc1 = nn.Linear(11, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, self.actions_num)

    @auto_move_data
    def forward(self, x):
        x = x.float()

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)

        return x

    def update_batch_size(self):
        memory_size = len(self.agent.replay_memory)
        self.cur_batch_size = min(self.batch_size, memory_size)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-5)

    def prepare_data(self):
        self.agent.move(None)
        self.update_batch_size()

    def training_step(self, batch, batch_idx):
        game_state = self.agent.get_board_state()
        output = self(game_state.unsqueeze(0))

        self.agent.move(output)
        self.update_batch_size()

        y, y_hat = batch

        loss = F.mse_loss(y, y_hat)

        return {'loss': loss}

    def train_dataloader(self):
        return DataLoader(self.agent.replay_memory, batch_size=self.cur_batch_size, collate_fn=self.custom_collate)

    def custom_collate(self, batch):
        state_batch = torch.cat(tuple(d[0].unsqueeze(0) for d in batch))
        action_batch = torch.cat(tuple(d[1].unsqueeze(0) for d in batch))
        reward_batch = tuple(d[2] for d in batch)
        next_state_batch = torch.cat(tuple(d[3].unsqueeze(0) for d in batch))
        terminal_batch = tuple(d[4] for d in batch)

        next_state_output = self(next_state_batch)

        y_hat = torch.cat(tuple(torch.tensor([reward if terminal else reward + self.gamma * torch.max(next_output)]) for reward, terminal, next_output in zip(reward_batch, terminal_batch, next_state_output)))
        y_hat.detach_()

        pred = self(state_batch)
        action_batch = action_batch.type_as(pred)
        y_hat = y_hat.type_as(pred)

        y = torch.sum(pred * action_batch, dim=1)

        return y, y_hat


if __name__ == '__main__':
    model = SnakeNet()
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=model.iterations_num
    )
    trainer.fit(model)

    torch.save(model.state_dict(), './model.pt')
