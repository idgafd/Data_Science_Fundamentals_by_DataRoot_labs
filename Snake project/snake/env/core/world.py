import numpy as np
import random

from settings.constants import DIRECTIONS, SNAKE_SIZE, DEAD_REWARD, \
    MOVE_REWARD, EAT_REWARD, FOOD_BLOCK, WALL
from env.core.snake import Snake


class World(object):
    def __init__(self, size, custom, start_position, start_direction_index, food_position):
        """
        @param size: tuple
        @param custom: bool
        @param start_position: tuple
        @param start_direction_index: int
        @param food_position: tuple
        """
        # for custom init
        self.custom = custom
        self.start_position = start_position
        self.start_direction_index = start_direction_index
        self.food_position = food_position
        self.current_available_food_positions = None
        # rewards
        self.DEAD_REWARD = DEAD_REWARD
        self.MOVE_REWARD = MOVE_REWARD
        self.EAT_REWARD = EAT_REWARD
        self.FOOD = FOOD_BLOCK
        self.WALL = WALL
        self.DIRECTIONS = DIRECTIONS
        # Init a numpy ndarray with zeros of predefined size - that will be the initial World
        self.size = size
        self.world = np.zeros(size, dtype=np.float64)
        # Fill in the indexes gaps to add walls along the World's boundaries
        self.world[0, :] = self.WALL
        self.world[-1, :] = self.WALL
        self.world[:, 0] = self.WALL
        self.world[:, -1] = self.WALL
        # Get available positions for placing food
        # Food should not to be spawned in the Walls
        self.available_food_positions = set(zip(*np.where(self.world == 0)))
        # Init snake
        self.snake = self.init_snake()
        # Set food
        self.init_food()

    def init_snake(self):
        """
        Initialize a snake
        """
        if not self.custom:
            # Choose a random position for spawn the Snake
            # Tail should not spawn outside of the box or in the wall
            # Remember, coordinates is a tuple(X, Y)
            start_position = random.choice(list(self.available_food_positions))
            # Choose a random direction index
            start_direction_index = random.randint(0, 3)
            new_snake = Snake(start_position, start_direction_index, SNAKE_SIZE)
        else:
            new_snake = Snake(self.start_position, self.start_direction_index, SNAKE_SIZE)
        return new_snake

    def init_food(self):
        """
        Initialize a piece of food
        """
        snake = self.snake if self.snake.alive else None
        # Update available positions for food placement considering snake location
        # Food should not be spawned in the Snake
        # self.current_available_food_positions should be the set
        self.current_available_food_positions = self.available_food_positions - set(self.snake.blocks)
        if not self.custom:
            # Choose a random position from available now
            chosen_position = random.choice(list(self.current_available_food_positions))
        else:
            chosen_position = self.food_position
            # Code needed for checking your project. Just leave it as it is
            try:
                self.current_available_food_positions.remove(chosen_position)
            except:
                if (self.food_position[0] - 1, self.food_position[1]) in self.current_available_food_positions:
                    chosen_position = (self.food_position[0] - 1, self.food_position[1])
                else:
                    chosen_position = (self.food_position[0] - 1, self.food_position[1] + 1)
                self.current_available_food_positions.remove(chosen_position)
        self.world[chosen_position[0], chosen_position[1]] = self.FOOD
        self.food_position = chosen_position

    def get_observation(self):
        """
        Get observation of current world state
        """
        obs = self.world.copy()
        snake = self.snake if self.snake.alive else None
        # Here we placing Snake on the World grid with SNAKE_BLOCKs
        if snake:
            for block in snake.blocks:
                obs[block[0], block[1]] = snake.snake_block
                # snakes head
            obs[snake.blocks[0][0], snake.blocks[0][1]] = snake.snake_block + 1
        return obs

    def move_snake(self, action):
        """
        Action executing
        """
        # define reward variable
        reward = 0
        # food needed flag
        new_food_needed = False
        # check if snake is alive
        if self.snake.alive:
            # perform a step (from Snake class)
            new_snake_head, old_snake_tail = self.snake.step(action)
            # Check if snake is outside bounds
            if (new_snake_head[0] <= 0 or new_snake_head[0] >= self.size[0] or
                    new_snake_head[1] <= 0 or new_snake_head[1] >= self.size[1]):
                self.snake.alive = False
            # Check if snake eats itself
            elif new_snake_head in self.snake.blocks[1:]:
                self.snake.alive = False
            # Check if snake eats the food
            if self.snake.alive and new_snake_head == self.food_position:
                # Remove old food
                self.world[self.food_position[0], self.food_position[1]] = 0
                # Add tail again
                # Note: all Snake coordinates should be tuples(X, Y)
                self.snake.blocks.append(old_snake_tail)
                # Request to place new food
                new_food_needed = True
                reward = self.EAT_REWARD
            elif self.snake.alive:
                # Didn't eat anything, move reward
                reward = self.MOVE_REWARD
        # Compute done flag and assign dead reward
        done = not self.snake.alive
        reward = reward if self.snake.alive else self.DEAD_REWARD
        # Adding new food
        if new_food_needed:
            self.init_food()
        return reward, done, self.snake.blocks
