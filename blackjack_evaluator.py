"""
Solving Blackjack with Q-Learning
=================================

"""




from __future__ import annotations
from typing import Tuple, SupportsFloat
import numpy as np
import gymnasium as gym

import pickle
import sys

from blackjack_utils import STATE_ACTION_SPACE_SHAPE
from blackjack_individual import BlackjackIndividual

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.fitness.simple_fitness import SimpleFitness


class BlackjackEvaluator(SimpleIndividualEvaluator):
    def __init__(self,
                 initial_epsilon: float,
                 epsilon_decay: float,
                 final_epsilon: float,
                 arity=1,
                 events=None,
                 event_names=None,
                 n_episodes=100_000):
        super().__init__(arity, events, event_names)
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.n_episodes = n_episodes

        # Dump this instance into a pickle file, which will later be used for evaluation
        with open('blackjack_evaluator.pkl', 'wb') as f:
            pickle.dump(self, f)

    def evaluate_individual(self, individual):
        # agent = BlackjackAgent(*individual.get_vector())
        vector = individual.get_vector()
        q_values = np.reshape(vector, STATE_ACTION_SPACE_SHAPE)

        env = gym.make("Blackjack-v1", sab=True)
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=self.n_episodes)
        
        epsilon = self.initial_epsilon
        rewards = np.zeros(STATE_ACTION_SPACE_SHAPE)

        for episode in range(self.n_episodes):
            obs, info = env.reset()
            done = False

            # play one episode
            while not done:
                # convert all observation values to integers
                obs = tuple(int(x) for x in obs)
                action = self.get_action(obs, env, q_values, epsilon)
                next_obs, reward, terminated, truncated, info = env.step(action)

                # update the rewards
                self.update(obs, action, reward, rewards)
                # agent.update(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

            epsilon = self.decay_epsilon(epsilon)

        env.close()
        individual.set_rewards(np.flatten(rewards))
        return np.sum(np.array(env.return_queue).flatten())

    def get_action(self, obs: Tuple[int, int, bool], env: gym.Env, q_values: np.ndarray, epsilon: float) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: SupportsFloat,
        rewards: np.ndarray
    ):
        rewards[obs][action] += reward

    def decay_epsilon(self, epsilon: float) -> float:
        return max(self.final_epsilon, epsilon - self.epsilon_decay)
    
def main():
    if len(sys.argv) == 1:
        print('Usage: python blackjack_evaluator.py <individual_vector_cells> (seperated by whitespaces)')
        sys.exit(1)

    # Initialize the evaluator
    with open('blackjack_evaluator.pkl', 'rb') as f:
        blackjack_evaluator = pickle.load(f)

    # Parse the given individual, then evaluate it
    vector = [float(cell) for cell in sys.argv[1:]]

    ind = BlackjackIndividual(SimpleFitness(), length=len(vector), bounds=(0.0, 1.0))
    ind.set_vector(vector)
    fitness = blackjack_evaluator.evaluate_individual(ind)

    # Write the fitness to stdout
    print(fitness, flush=True)

if __name__ == '__main__':
    main()

