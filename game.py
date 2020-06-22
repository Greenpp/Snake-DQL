from random import randint
import numpy as np
from matplotlib import pyplot as plt
from time import sleep


class Engine:
    def __init__(self, board_size=40, round_time=.5):
        self.board_size = board_size
        self.round_time = round_time

        self.init_snake()
        self.spawn_fruit()

        self.points = 0
        self.round = 1

    def to_numpy(self):
        board = np.zeros((self.board_size, self.board_size))
        for s in self.snake:
            x, y = s
            board[x, y] = -1

        fx, fy = self.fruit
        board[fx, fy] = 1

        return board
    
    def change_direction(self, direction):
        if direction == 'up':
            d = 0
        elif direction == 'right':
            d = 1
        elif direction == 'down':
            d = 2
        else:
            d = 3

        if self.snake_direction % 2 != d % 2:
            self.snake_direction = d

    def init_snake(self):
        self.alive = True

        pos = self.board_size // 2
        self.snake = []
        self.snake.append([pos, pos])

        self.snake_direction = randint(0, 3)
        # 0 - top, 1 - right, 2 - down, 3 - left

    def spawn_fruit(self):
        pos = self.snake[0]
        while pos in self.snake:
            pos = [randint(0, self.board_size - 1),
                   randint(0, self.board_size - 1)]

        self.fruit = pos

    def check_if_alive(self):
        head = self.snake[0]
        if head in self.snake[1:]:
            self.alive = False
        elif head[0] < 0 or head[0] > self.board_size - 1:
            self.alive = False
        elif head[1] < 0 or head[1] > self.board_size - 1:
            self.alive = False

    def next_round(self):
        head = self.snake[0].copy()

        if self.snake_direction == 0:
            head[1] -= 1
        elif self.snake_direction == 1:
            head[0] += 1
        elif self.snake_direction == 2:
            head[1] += 1
        else:
            head[0] -= 1

        self.snake.append(head)

        if self.snake[0] != self.fruit:
            del self.snake[-1]
        else:
            self.spawn_fruit()


if __name__ == '__main__':
    eng = Engine()

    print(eng.to_numpy())