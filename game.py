from random import randint

import numpy as np


class Engine:
    def __init__(self, board_size=40):
        self.board_size = board_size

        self.food_reward = 1
        self.dead_penalty = 1
        self.round_penalty = .01

        self.reset()

    def reset(self):
        self.init_snake()
        self.spawn_fruit()

        self.last_round_points = 0
        self.points = 0
        self.round = 1

    def init_snake(self):
        self.alive = True

        pos = self.board_size // 2
        self.snake = []
        self.snake.append([pos, pos])

        self.snake_direction = randint(0, 3)
        self.snake_new_direction = self.snake_direction
        # 0 - top, 1 - right, 2 - down, 3 - left

    def spawn_fruit(self):
        pos = self.snake[0]
        while pos in self.snake:
            pos = [randint(0, self.board_size - 1),
                   randint(0, self.board_size - 1)]

        self.fruit = pos

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
            self.snake_new_direction = d

    def check_if_alive(self):
        head = self.snake[-1]
        if head in self.snake[:-1]:
            self.alive = False
        elif self.is_head_outside_board():
            self.alive = False

        if not self.alive:
            self.points -= self.dead_penalty

    def is_head_outside_board(self, head=None):
        if head is None:
            head = self.snake[-1]
        outside = False

        if head[0] < 0 or head[0] >= self.board_size:
            outside = True
        elif head[1] < 0 or head[1] >= self.board_size:
            outside = True

        return outside

    def check_food(self):
        head = self.snake[-1]

        if head != self.fruit:
            del self.snake[0]
        else:
            self.points += self.food_reward
            self.spawn_fruit()

    def move(self):
        self.snake_direction = self.snake_new_direction

        head = self.move_element(self.snake[-1], self.snake_direction)

        self.snake.append(head)

    def next_round(self):
        self.round += 1
        self.move()
        self.last_round_points = self.points
        self.points -= self.round_penalty

        self.check_if_alive()
        self.check_food()

    def next_round_nn(self, action):
        # action is -1 - turn left, 0 - nothing, 1 - turn right
        action -= 1
        self.snake_new_direction = (self.snake_direction + action) % 4
        self.next_round()

        board = self.get_board_state()
        reward = self.points - self.last_round_points
        terminal = not self.alive

        return board, reward, terminal

    def move_element(self, element, direction):
        m_element = element.copy()

        if direction == 0:
            m_element[1] += 1
        elif direction == 1:
            m_element[0] += 1
        elif direction == 2:
            m_element[1] -= 1
        else:
            m_element[0] -= 1

        return m_element

    def get_board_state(self):
        head = self.snake[-1]

        # Direction
        dir_state = np.zeros(4)
        dir_state[self.snake_direction] = 1

        # Danger
        danger_state = np.zeros(3)

        tmp_direction = self.snake_direction
        tmp_head = self.move_element(head, tmp_direction)
        if tmp_head in self.snake[:-1] or self.is_head_outside_board(tmp_head):
            danger_state[0] = 1

        tmp_direction = (self.snake_direction + 1) % 4
        tmp_head = self.move_element(head, tmp_direction)
        if tmp_head in self.snake[:-1] or self.is_head_outside_board(tmp_head):
            danger_state[1] = 1

        tmp_direction = (self.snake_direction - 1) % 4
        tmp_head = self.move_element(head, tmp_direction)
        if tmp_head in self.snake[:-1] or self.is_head_outside_board(tmp_head):
            danger_state[2] = 1

        # Food
        food_state = np.zeros(4)
        fx, fy = self.fruit
        hx, hy = head

        if fx < hx:
            food_state[0] = 1
        if fx > hx:
            food_state[1] = 1
        if fy < hy:
            food_state[2] = 1
        if fy > hy:
            food_state[3] = 1

        return np.concatenate((dir_state, danger_state, food_state))

    def to_numpy(self):
        board = np.zeros((2, self.board_size + 2, self.board_size + 2))
        border = np.pad(np.zeros((self.board_size, self.board_size)),
                        1, constant_values=1)[np.newaxis, :]
        board = np.concatenate((border, board), axis=0)
        *body, head = self.snake
        for s in body:
            x, y = s
            x += 1
            y += 1

            board[0, x, y] = 1

        x, y = head
        if 0 < x < self.board_size - 1 and 0 < y < self.board_size - 1:
            board[1, x+1, y+1] = 1
            board[0, x+1, y+1] = 1

        x, y = self.fruit
        board[2, x+1, y+1] = 1

        return board


if __name__ == '__main__':
    eng = Engine(board_size=10)

    print(eng.get_board_state().shape)
