import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.core.decorators import auto_move_data
from torch.utils.data import DataLoader, IterableDataset

from game import Engine


class Memory:
    """
    Memory holding given number of last experiences with ability to sample it randomly
    """

    def __init__(self, size):
        self.max_size = size
        self.buffer = []

    def sample(self, num):
        """
        Sample <num> random experiences
        """
        buff_size = len(self.buffer)
        sample_size = min(num, buff_size)
        sample = random.sample(self.buffer, sample_size)

        return sample

    def append(self, exp):
        """
        Add new experience and remove the oldest if capacity is reached
        """
        self.buffer.append(exp)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)


class RLDataset(IterableDataset):
    """
    Dataset serving stored in memory experiences
    """

    def __init__(self, memory, iterations_num, batch_size):
        super(RLDataset).__init__()
        self.memory = memory
        self.iterations_num = iterations_num
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.iterations_num):
            data = self.memory.sample(self.batch_size)
            while data:
                yield data.pop()


class Agent:
    """
    Reinforcement learning agent interacting with the game environment
    """

    def __init__(self, model, memory_size, iterations_num, batch_size, head_surrounding_size, game_board_size=20, random_start=True):
        self.model = model
        self.head_surrounding_size = head_surrounding_size
        self.game_engine = Engine(game_board_size, random_start)
        self.replay_memory = Memory(memory_size)

        self.dataset = RLDataset(
            self.replay_memory, iterations_num, batch_size)

        self.game_state = self.get_game_state()

    def get_dataset(self):
        """
        Returns dataset based on replay memory
        """

        return self.dataset

    def get_game_state(self):
        game_state = self.game_engine.get_game_state()
        hs_size = self.head_surrounding_size * 2 + 1
        head_state = self.game_engine.get_head_surrounding(
            self.head_surrounding_size) if self.game_engine.alive else np.zeros((hs_size, hs_size))

        game_state = torch.from_numpy(game_state).unsqueeze(0)
        head_state = torch.from_numpy(head_state).unsqueeze(0).unsqueeze(0)

        state = (head_state, game_state)

        return state

    def move(self, epsilon):
        """
        Make single interaction with the game environment
        """
        actions_num = self.model.actions_num

        rnd_action = random.random() <= epsilon
        if rnd_action:
            action_idx = random.randint(0, actions_num - 1)
        else:
            model_output = self.model(self.game_state)
            action_idx = torch.argmax(model_output).item()
        action = torch.zeros(actions_num)
        action[action_idx] = 1
        action = action.unsqueeze(0)

        direction = self.translate_action(action_idx)
        reward, terminal = self.game_engine.next_round(direction)
        new_state = self.get_game_state()

        exp = (self.game_state, action, reward, new_state, terminal)
        self.replay_memory.append(exp)

        if terminal:
            self.game_engine.reset()
            new_state = self.get_game_state()
        self.game_state = new_state

    def translate_action(self, action):
        """
        Translate action to new direction

        Actions:
        0 - turn left
        1 - go straight
        2 - turn right

        """
        direction = self.game_engine.direction
        new_direction = (direction + action - 1) % 4

        return new_direction

    def warmup(self, num):
        """
        Make <num> of random moves
        """
        for _ in range(num):
            self.move(1)


class SnakeNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.actions_num = 3
        self.head_surrounding_size = 4
        self.head_state_features = 16
        self.game_features = 11

        self.gamma = .95
        self.epsilon_init = 1
        self.epsilon_final = .0001
        self.iterations_num = 20000
        self.replay_memory_size = 10000
        self.batch_size = 32
        self.warmup_rounds = self.batch_size
        self.board_size = 10
        self.random_start = False

        self.epsilon = self.epsilon_init
        self.epsilon_delta = (self.epsilon_init -
                              self.epsilon_final) / self.iterations_num

        self.agent = Agent(self, self.replay_memory_size, self.iterations_num,
                           self.batch_size, self.head_surrounding_size, self.board_size,
                           self.random_start)
        self.net = nn.Sequential(
            nn.Linear(self.game_features + self.head_state_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.actions_num),
            nn.Softmax(dim=1)
        )

        self.head_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                pow(2 * self.head_surrounding_size + 1, 2) * 64,
                self.head_state_features
            ),
            nn.Sigmoid()
        )

    @auto_move_data
    def forward(self, x):
        head_data, state_data = x
        head_data = head_data.float()
        state_data = state_data.float()

        head_state = self.head_net(head_data)
        x = torch.cat((head_state, state_data), dim=1)

        return self.net(x)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-5)
        sch = optim.lr_scheduler.StepLR(opt, 5000, .1)
        return [opt], [sch]

    def prepare_data(self):
        return self.agent.warmup(self.warmup_rounds)

    def update_epsilon(self):
        self.epsilon -= self.epsilon_delta

    def training_step(self, batch, batch_idx):
        self.agent.move(self.epsilon)
        self.update_epsilon()

        batch = self.process_batch(batch)

        y, y_hat = batch
        loss = F.mse_loss(y, y_hat)

        return {'loss': loss}

    def train_dataloader(self):
        ds = self.agent.get_dataset()
        dl = DataLoader(ds, batch_size=self.batch_size,
                        collate_fn=lambda x: x)

        return dl

    def process_batch(self, batch):
        head_state = torch.cat(tuple(d[0][0] for d in batch))
        game_state = torch.cat(tuple(d[0][1] for d in batch))
        state_batch = (head_state, game_state)

        action_batch = torch.cat(tuple(d[1] for d in batch))
        reward_batch = tuple(d[2] for d in batch)

        new_head_state = torch.cat(tuple(d[3][0] for d in batch))
        new_game_state = torch.cat(tuple(d[3][1] for d in batch))
        new_state_batch = (new_head_state, new_game_state)

        terminal_batch = tuple(d[4] for d in batch)

        with torch.no_grad():
            new_state_output = self(new_state_batch)
            y_hat = torch.tensor(tuple([reward if terminal else reward + self.gamma * torch.max(new_output)]
                                       for reward, terminal, new_output in zip(reward_batch, terminal_batch, new_state_output)))
            y_hat = y_hat.squeeze()

        pred = self(state_batch)
        action_batch = action_batch.type_as(pred)
        y_hat = y_hat.type_as(pred)

        y = torch.sum(pred * action_batch, dim=1)

        return y, y_hat


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    model = SnakeNet()
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1
    )
    trainer.fit(model)
    trainer.save_checkpoint('model.ptl')
