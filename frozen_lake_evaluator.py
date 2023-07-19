'''
Origin: Solving Blackjack with Q-Learning
https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
'''


import numpy as np
import gymnasium as gym
import pickle
import sys
import json
from typing import List, SupportsFloat

import utils
import multiprocessing
from frozen_lake_individual import FrozenLakeIndividual

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.fitness.simple_fitness import SimpleFitness


class FrozenLakeEvaluator(SimpleIndividualEvaluator):
    def __init__(self,
                 arity=1,
                 events=None,
                 event_names=None,
                 use_rewards=False,
                 total_episodes=1000,
                 is_slippery=True):
        super().__init__(arity, events, event_names)
        self.total_episodes = total_episodes
        self.use_rewards = use_rewards
        if use_rewards:
            self.ids2rewards = multiprocessing.Manager().dict()

        # Generate a random 8x8 map with 80% of the cells being frozen
        # This map will remain the same through the whole evolutionary run
        map_size = utils.FROZEN_LAKE_MAP_SIZE
        self.env = gym.make('FrozenLake-v1',
                            map_name=f'{map_size}x{map_size}',
                            is_slippery=is_slippery)

        # Dump this instance into a pickle file, which will later be used for evaluation
        with open('frozen_lake_evaluator.pkl', 'wb') as f:
            pickle.dump(self, f)

    def evaluate_individual(self, individual):
        vector = individual.get_vector()
        rewards = np.zeros(len(vector)) if self.use_rewards else None

        for hole in utils.HOLES:
            vector.insert(hole, 0)

        score = 0
        for episode in range(self.total_episodes):
            state = self.env.reset()[0]  # Reset the environment
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = self.choose_action(state=state, vector=vector)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated

                if self.use_rewards:
                    self.update(state, action, reward, rewards)

                total_rewards += reward
                step += 1

                # Our new state is state
                state = new_state
            
            score += total_rewards
        
        if self.use_rewards:
            self.ids2rewards[individual.id] = rewards
        
        return score / self.total_episodes

    def choose_action(self, state: int, vector: List[int]) -> int:
        """
        Returns the best action.
        """
        return int(vector[state])
    
    def update(
        self,
        state: int,
        action: int,
        reward: SupportsFloat,
        rewards: np.ndarray
    ):
        if reward:
            # Big reward to an action that leads to the goal
            rewards[state] += reward * self.total_episodes
        else:
            # Small penalty to each step that doesn't lead to the goal
            rewards[state] -= 1 / self.total_episodes

    def terminate(self):
        self.env.close()
