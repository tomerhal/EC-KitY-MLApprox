"""
Test dynamic model creation and usage.
"""
import pytest
from eckity.genetic_encodings.ga.vector_individual import Vector
from eckity.fitness.simple_fitness import SimpleFitness

import sys
sys.path.append('..')
from plato_termination_checker import PlatoTerminationChecker

class TestPlato:
    def _should_terminate(self, gens, threshold, higher_is_better, fitness_history, curr_fitness):
        term_checker = PlatoTerminationChecker(gens=gens, threshold=threshold, higher_is_better=higher_is_better)
        term_checker.fitness_history = fitness_history
        best_ind = Vector(SimpleFitness(curr_fitness), bounds=(-1, 1))
        return term_checker.should_terminate(None, best_ind, gens)

    def test_should_terminate_maximization_success(self):
        gens = 3
        threshold = 0.1
        higher_is_better = True
        fitness_history = [0.9] * gens
        curr_fitness = 1.0
        assert self._should_terminate(gens, threshold, higher_is_better, fitness_history, curr_fitness)

    def test_should_terminate_maximization_fail(self):
        gens = 3
        threshold = 0.1
        higher_is_better = True
        fitness_history = [0.9] * gens
        curr_fitness = 0.98
        assert not self._should_terminate(gens, threshold, higher_is_better, fitness_history, curr_fitness)
        

