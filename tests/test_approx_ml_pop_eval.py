"""
Test dynamic model creation and usage.
"""
import pytest
import random
import numpy as np

from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from eckity.genetic_encodings.ga.vector_individual import Vector
from eckity.fitness.simple_fitness import SimpleFitness

import sys
sys.path.append('..')
from approx_ml_pop_eval import ApproxMLPopulationEvaluator


class TestApproxMLPopulationEvaluator:
	def test_model(self):
		models_params = {
			SGDRegressor: {'max_iter': 1000, 'tol': 1e-3, 'random_state': 0},
			LinearRegression: {},
			KNeighborsRegressor: {'n_neighbors': 3}
		}

		X_train = [Vector(SimpleFitness(random.random()), (i, i+1)) for i in range(10)]
		for i, ind in enumerate(X_train):
			ind.set_vector([i, i+1])

		y_train = [ind.fitness.fitness for ind in X_train]
		
		X_test = [Vector(SimpleFitness(random.random()), (i+10, i+11)) for i in range(2)]
		for i, ind in enumerate(X_test):
			ind.set_vector([i+10, i+11])
		y_test = [ind.fitness.fitness for ind in X_test]

		for model_type, params in models_params.items():
			pop_eval = ApproxMLPopulationEvaluator(model_type=model_type, model_params=params)
			pop_eval.fit(X_train, y_train)

			model = model_type(**params)
			V_train = np.array([ind.vector for ind in X_train])
			model.fit(V_train, y_train)

			V_test = np.array([ind.vector for ind in X_test])
			assert model.score(V_test, y_test) == pytest.approx(pop_eval.model.score(V_test, y_test))
		