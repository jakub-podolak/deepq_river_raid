import os
import random 
from pathlib import Path

import pygame
import numpy as np
import cv2

from actions import LEFT, NONE, RIGHT

STATE_WIDTH = 80
STATE_HEIGHT = 80

SCREEN_WIDTH = 320
SCREEN_HEIGHT = 320

MAPS_PATH = Path('maps/')

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class Environment(object):

    def read_random_map(self):
        selected_map = random.choice(os.listdir(MAPS_PATH))
        
        self.next_map_row = 0
        self.loaded_map = cv2.imread(str(MAPS_PATH / selected_map))

        # BGR to RGB
        self.loaded_map = self.loaded_map[...,::-1]
        return map
        

    def move_board_one_line(self):
        if self.next_map_row == self.loaded_map.shape[0]:
            self.read_random_map()

        new_line = self.loaded_map[self.next_map_row]

        self.current_board = np.concatenate(([new_line], self.current_board[:-1]))
        self.next_map_row += 1


    def reset_state(self):
        # initialize map
        self.read_random_map()
        self.current_board = np.zeros((STATE_HEIGHT, STATE_WIDTH, 3), dtype='uint8')

        for _ in range(SCREEN_HEIGHT):
            self.move_board_one_line()

        # initialize player
        self.player_x = STATE_WIDTH // 2
        self.player_y = STATE_HEIGHT - 10


    def __init__(self):
        # initialize pygame
        pygame.init() 
        self.screen = pygame.display.set_mode((SCREEN_HEIGHT, SCREEN_WIDTH))
        self.clock = pygame.time.Clock()

        self.reset_state()
        

    def set_window_name(self, window_name):
        pygame.display.set_caption(window_name)


    def cvimage_to_pygame(self, image):
        """Convert cvimage into a pygame image"""
        return pygame.image.frombuffer(image.tostring(), image.shape[1::-1], "RGB")


    def get_player_shape(self):
        return [
            (self.player_y, self.player_x),
            (self.player_y + 1, self.player_x),
            (self.player_y, self.player_x + 1),
            (self.player_y + 1, self.player_x + 1)
        ]


    def get_current_state(self):
        state = self.current_board.copy()
        player_points = self.get_player_shape()

        for pp in player_points:
            state[pp[0]][pp[1]] = RED

        return state


    def render(self):
        current_state = self.get_current_state()
        to_show = cv2.resize(current_state, (SCREEN_HEIGHT, SCREEN_WIDTH), interpolation=cv2.INTER_AREA)
        self.screen.blit(self.cvimage_to_pygame(to_show), (0, 0))
        pygame.display.update()


    def check_collision(self):
        player_points = self.get_player_shape()

        for pp in player_points:
            if (self.current_board[pp[0]][pp[1]] == GREEN).all():
                return True

        return False


    def handle_keys(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            return LEFT
        if keys[pygame.K_RIGHT]:
            return RIGHT


    def step(self, action):
        if action == LEFT:
            self.player_x -= 1
            if self.player_x == -1:
                self.player_x = 0
        
        if action == RIGHT:
            self.player_x += 1
            if self.player_x == STATE_WIDTH - 1:
                self.player_x = STATE_WIDTH - 2

        self.move_board_one_line()

        if self.check_collision():
            return self.get_current_state(), 0, True
        
        return self.get_current_state(), 0.1, False


    def get_state_shape(self):
        return self.current_board.shape


    def close_window(self):
        pygame.quit()


    def should_close(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

    
    def frame_sleep(self, fps=30):
        self.clock.tick(fps)