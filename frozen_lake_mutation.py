import numpy as np
from random import choices

from eckity.genetic_operators.mutations.vector_n_point_mutation \
    import VectorNPointMutation
from blackjack_individual import BlackjackIndividual


class FrozenLakeMutation(VectorNPointMutation):
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
