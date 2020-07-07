from random import randint

import numpy as np


class Engine:
    """
    Snake game engine

    Used for manageing game environment
    """

    def __init__(self, board_size):
        self.board_size = board_size

        self.alive = None
        self.direction = None
        self.fruit = None
        self.points = None
        self.round = None
        self.snake = None

        self.food_reward = 1
        self.death_penalty = 1
        self.round_penalty = .01

        self.reset()

    def reset(self):
        """
        Reset environment to initial state
        """
        self.points = 0
        self.round = 1

        self._reset_snake()
        self._reset_fruit()

    def _reset_snake(self):
        """
        Reset snake to initial state
        """
        self.alive = True
        self.snake = []

        init_pos = self.board_size // 2
        self.snake.append([init_pos, init_pos])

        self.direction = randint(0, 3)

    def _reset_fruit(self):
        """
        Respawn fruit
        """
        pos = self.snake[0]
        while pos in self.snake:
            pos = [randint(0, self.board_size - 1),
                   randint(0, self.board_size - 1)]

        self.fruit = pos

    def _is_pos_outside(self, pos):
        """
        Check if give position is outside the board
        """
        x, y = pos
        is_outside = False
        if x < 0 or x >= self.board_size:
            is_outside = True
        elif y < 0 or y >= self.board_size:
            is_outside = True

        return is_outside

    def _has_died(self):
        """
        Check if snake collided with wall or itself
        """
        *body, head = self.snake

        return head in body or self._is_pos_outside(head)

    def _has_eaten(self):
        """
        Check if snake found fruit
        """
        head = self.snake[-1]

        return head == self.fruit

    def _move_pos(self, pos, direction):
        """
        Get new position after move in given direction
        """
        new_pos = pos.copy()

        if direction == 0:
            new_pos[1] += 1
        elif direction == 1:
            new_pos[0] += 1
        elif direction == 2:
            new_pos[1] -= 1
        else:
            new_pos[0] -= 1

        return new_pos

    def _move(self):
        """
        Move snake
        """
        head = self.snake[-1]

        new_head = self._move_pos(head, self.direction)
        self.snake.append(new_head)

    def next_round(self, direction):
        """
        Transfer game into next round
        """
        self.round += 1
        self.direction = direction

        self._move()

        reward = 0
        reward -= self.round_penalty
        if self._has_eaten():
            reward += self.food_reward
            self._reset_fruit()
        else:
            self.snake.pop(0)
        if self._has_died():
            self.alive = False
            reward -= self.death_penalty

        self.points += reward

        return reward, self.alive

    def get_game_state(self):
        """
        Get binary vector describing current game state
        """
        tail, *body, head = self.snake

        # Direction
        dir_state = np.zeros(4)
        dir_state[self.direction] = 1

        # Food
        food_state = np.zeros(4)
        fx, fy = self.fruit
        hx, hy = head

        if fx < hx:
            food_state[0] = 1
        elif fx > hx:
            food_state[1] = 1
        if fy < hy:
            food_state[2] = 1
        elif fy > hy:
            food_state[3] = 1

        # Danger
        danger_state = np.zeros(3)

        dirs = [-1, 0, 1]
        for i, d in enumerate(dirs):
            tmp_dir = (self.direction + d) % 4
            tmp_head = self._move_pos(head, tmp_dir)
            if tmp_head in body or self._is_pos_outside(tmp_head):
                danger_state[i] = 1

        return np.concatenate([dir_state, food_state, danger_state])

        
