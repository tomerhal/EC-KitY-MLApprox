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
import random

import utils
import multiprocessing
from frozen_lake_individual import FrozenLakeIndividual

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.fitness.simple_fitness import SimpleFitness


class MyEvaluator(SimpleIndividualEvaluator):
    def __init__(self,
                 arity=1,
                 events=None,
                 event_names=None):
        super().__init__(arity, events, event_names)
        self.ids2rewards = multiprocessing.Manager().dict()

    def evaluate_individual(self, individual):
        return random.random()
