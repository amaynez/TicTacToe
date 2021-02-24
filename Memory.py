import random


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position % self.capacity] = experience
        self.position += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_sample_memory(self, batch_size):
        return len(self.memory) >= batch_size
