from eckity.genetic_encodings.ga.float_vector import FloatVector
from eckity.fitness.simple_fitness import SimpleFitness

from typing import Tuple
import numpy as np

class BlackjackIndividual(FloatVector):
    def __init__(
        self,
        fitness: SimpleFitness,
        length: int,
        bounds: Tuple[int, int]
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        super().__init__(fitness, length, bounds)
        self.rewards = None

    def set_rewards(self, rewards):
        self.rewards = rewards