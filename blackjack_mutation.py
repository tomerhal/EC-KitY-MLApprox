import numpy as np
from random import random, choices

from typing import List

from eckity.genetic_operators.mutations.vector_n_point_mutation import VectorNPointMutation
from blackjack_individual import BlackjackIndividual

class BlackjackMutation(VectorNPointMutation):
    def __init__(self, n=1, probability=1.0, probability_for_each=1.0, events=None):
        self.probability_for_each = probability_for_each
        super().__init__(n=n,
                         probability=probability,
                         arity=1,
                         cell_selector=self.sample_cells_by_rewards,
                         mut_val_getter=lambda individual, index: individual.bit_flip(
                             index) if random() <= self.probability_for_each else individual.cell_value(index),
                         events=events)
        
    def sample_cells_by_rewards(self, individual: BlackjackIndividual):
        vector_indices = range(individual.size())
        rewards = individual.get_rewards()
        rewards += np.abs(np.min(rewards)) + 1
        inverse_rewards = 1 / rewards

        # currently this will select with replacements
        return choices(vector_indices, weights=inverse_rewards, k=self.n)
