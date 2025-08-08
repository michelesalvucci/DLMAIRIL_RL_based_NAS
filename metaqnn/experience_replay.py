from collections import deque
import random
from typing import Dict, List

from models.layers import AbstractLayer


class ExperienceReplay:
    """
    Experience replay buffer for storing and sampling architecture experiences.
    Stores (architecture, validation_accuracy) pairs to accelerate learning.
    """
    def __init__(self, max_size=1500):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add_experience(self, architecture: List[AbstractLayer], reward: float):
        """Add an architecture experience to the replay buffer"""
        experience = {
            'architecture': architecture.copy(),
            'reward': reward
        }
        self.buffer.append(experience)
    
    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample a batch of experiences from the replay buffer"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)

    def size(self) -> int:
        """Return the current size of the replay buffer"""
        return len(self.buffer)