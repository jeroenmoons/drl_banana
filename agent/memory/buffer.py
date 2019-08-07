import random
from collections import namedtuple, deque


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples. Used to sample past experiences to learn from."""

    def __init__(self, action_size, buffer_size):
        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, sample_size):
        """Select a random batch of experiences from the buffer."""
        memory_size = len(self.buffer)
        sample_size = sample_size if sample_size < memory_size else memory_size

        return random.sample(self.buffer, k=sample_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)
