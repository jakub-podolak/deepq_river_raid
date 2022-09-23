from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def learn_from_past(self):
        ...
        
    @abstractmethod
    def train(self, environment):
        ...


    @abstractmethod
    def evaluate(self, state):
        ... 


    @abstractmethod
    def get_name(self):
        ... 


    @abstractmethod
    def plot(self):
        ...