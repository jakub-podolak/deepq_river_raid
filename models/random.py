import random

import matplotlib.pyplot as plt

from models.base import Base
from actions import LEFT, NONE, RIGHT


class RandomModel(Base):
    def __init__(self, env, hyperparams):
        self.env = env 
        pass
    

    def learn_from_past(self):
        pass


    def train(self):
        pass

    
    def evaluate(self, state):
        possible_actions = [LEFT, NONE, RIGHT]
        return random.choice(possible_actions)


    def get_name(self):
        return "RANDOM MODEL"

    
    def plot(self):
        pass