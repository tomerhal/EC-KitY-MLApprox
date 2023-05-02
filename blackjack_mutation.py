import numpy as np
from random import choices

from typing import List

from eckity.genetic_operators.mutations.vector_n_point_mutation import VectorNPointMutation
from blackjack_individual import BlackjackIndividual




class BlackjackUniformMutation(VectorNPointMutation):
    def __init__(self, n=1, probability=1.0, events=None):
        super().__init__(n=n,
                         probability=probability,
                         arity=1,
                         cell_selector=self.sample_cells_by_rewards,
                         mut_val_getter=lambda vec, index: vec.get_random_number_in_bounds(index),
                         events=events)
        
    def sample_cells_by_rewards(self, individual: BlackjackIndividual):
        vector_indices = range(individual.size())
        rewards = individual.get_rewards()
        rewards += np.abs(np.min(rewards)) + 1
        inverse_rewards = 1 / rewards

        # currently this will select with replacements
        return choices(vector_indices, weights=inverse_rewards, k=self.n)


class BlackjackGaussMutation(VectorNPointMutation):
    def __init__(self, n=1, probability=1.0, mu=0.0, sigma=1.0, events=None, attempts=5):
        super().__init__(n=n,
                         probability=probability,
                         arity=1,
                         cell_selector=self.sample_cells_by_rewards,
                         mut_val_getter=lambda vec, index: vec.get_random_number_with_gauss(index, mu, sigma),
                         events=events,
                         attempts=attempts)

    def sample_cells_by_rewards(self, individual: BlackjackIndividual):
        vector_indices = range(individual.size())
        rewards = individual.get_rewards()
        rewards += np.abs(np.min(rewards)) + 1
        choices(vector_indices, weights=individual.get_rewards(), k=self.n)

    def on_fail(self, payload):
        """
        Handle gauss mutation failure by invoking a uniform mutation
        with the same probability of the Gauss mutation.
        """
        mut = BlackjackUniformMutation(self.n, self.probability, self.arity, self.events)
        return mut.apply_operator(payload)