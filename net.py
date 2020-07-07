import random

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
    Memory holding given number of last experiences with ability to sample it random
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

    def __init__(self, memory, sample_size):
        super(RLDataset).__init__()
        self.memory = memory
        self.sample_size = sample_size

    def __iter__(self):
        data = self.memory.sample(self.sample_size)

        for d in data:
            yield d


class Agent:
    """
    Reinforcement learning agent interacting with the game environment
    """

    def __init__(self, model, memory_size, epoch_samples_num, game_board_size=20):
        self.model = model
        self.game_engine = Engine(game_board_size)
        self.replay_memory = Memory(memory_size)
        self.epoch_samples = epoch_samples_num

        self.game_state = self.game_engine.get_game_state()
        self.game_state = torch.from_numpy(self.game_state).unsqueeze(0)

    def get_dataset(self):
        """
        Create dataset using replay memory
        """
        ds = RLDataset(self.replay_memory, self.epoch_samples)

        return ds

    def move(self, epsilon):
        """
        Make single interaction with the game environment
        """
        actions_num = self.model.actions_num

        rnd_action = random.random() <= epsilon
        if rnd_action:
            action_idx = random.randint(0, actions_num - 1)
        else:
            model_output = self.model(game_state)
            action_idx = torch.argmax(model_output).item()
        action = torch.zeros(actions_num)
        action[action_idx] = 1
        action = action.unsqueeze(0)

        direction = self.translate_action(action_idx)
        reward, terminal = self.game_engine.next_round(direction)
        new_state = self.game_engine.get_game_state()
        new_state = torch.from_numpy(new_state).unsqueeze(0)

        exp = (self.game_state, action, reward, new_state, terminal)
        self.replay_memory.append(exp)

        if terminal:
            self.game_engine.reset()

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
        self.gamma = .9
        self.epsilon_init = 1
        self.epsilon_final = .0001
        self.iterations_num = 1000
        self.replay_memory_size = 10000
        self.batch_size = 32

        self.epsilon = self.epsilon_init
        self.epsilon_delta = (self.epsilon_final -
                              self.epsilon_init) / self.iterations_num

        self.agent = Agent(self, self.replay_memory_size, 1000)
        self.net = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.actions_num),
            nn.Softmax(dim=1)
        )

    @auto_move_data
    def forward(self, x):
        x = x.float()

        return self.net(x)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-5)
        return opt

    def prepare_data(self):
        return self.agent.warmup(1000)

    def update_epsilon(self):
        self.epsilon -= self.epsilon_delta

    def training_step(self, batch, batch_idx):
        self.agent.move(self.epsilon)
        self.update_epsilon()

        y, y_hat = batch
        loss = F.mse_loss(y, y_hat)

        return {'loss': loss}

    def train_dataloader(self):
        ds = self.agent.get_dataset()
        dl = DataLoader(ds, batch_size=self.batch_size,
                        collate_fn=self.custom_collate)

        return dl

    def custom_collate(self, batch):
        state_batch = torch.cat(tuple(d[0] for d in batch))
        action_batch = torch.cat(tuple(d[1] for d in batch))
        reward_batch = tuple(d[2] for d in batch)
        new_state_batch = torch.cat(tuple(d[3] for d in batch))
        terminal_batch = tuple(d[4] for d in batch)

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
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model)
    trainer.save_checkpoint('model.ptl')
