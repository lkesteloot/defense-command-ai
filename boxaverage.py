
from collections import deque

# Compute the box (straight) average of the previous "size" numbers.
class BoxAverage:
    def __init__(self, size):
        self.buffer = deque([], size)
        self.sum = 0

    def append(self, value):
        if len(self.buffer) == self.buffer.maxlen:
            self.sum -= self.buffer[0]
        self.sum += value
        self.buffer.append(value)

    def average(self):
        return self.sum / len(self.buffer) if len(self.buffer) != 0 else 0
