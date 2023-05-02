import numpy as np

from eckity.genetic_encodings.ga.float_vector import FloatVector
from eckity.fitness.simple_fitness import SimpleFitness

from typing import Tuple

class BlackjackIndividual(FloatVector):
    def __init__(
        self,
        fitness: SimpleFitness,
        length: int,
        bounds: Tuple[int, int]
    ):
        super().__init__(fitness, length, bounds)
        self.rewards = None

    def reset_rewards(self) -> None:
        self.rewards = np.ones(self.length)

    def set_rewards(self, rewards: np.ndarray) -> None:
        self.rewards = rewards

    def get_rewards(self) -> np.ndarray:
        return self.rewards